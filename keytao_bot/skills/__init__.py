"""
Skills loader for AI chat
加载和管理 AI skills
"""
import os
import re
import importlib.util
from pathlib import Path
from typing import Dict, List, Callable, Optional
from nonebot.log import logger

from ..utils.pending_confirmation import expand_pending_confirmation_copy


class SkillsManager:
    """Manage and load skills from skills directory"""
    
    def __init__(self, skills_dir: Optional[str] = None):
        if skills_dir is None:
            # Default to keytao_bot/skills/ (parent is keytao_bot/)
            current_dir = Path(__file__).parent
            self.skills_dir = current_dir
        else:
            self.skills_dir = Path(skills_dir)
        
        self.tools: List[Dict] = []
        self.tool_functions: Dict[str, Callable] = {}
        self.skill_docs: Dict[str, str] = {}
        # tool name -> owning skill, so a later skill cannot shadow an earlier one
        self.tool_owners: Dict[str, str] = {}
        # tool name -> its OpenAI function-calling schema, for argument validation
        self.tool_schemas: Dict[str, Dict] = {}


    def load_all_skills(self):
        """Load all skills from skills directory"""
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return
        
        for skill_path in self.skills_dir.iterdir():
            if skill_path.is_dir() and not skill_path.name.startswith('.'):
                self.load_skill(skill_path)
    
    def _parse_skill_md(self, content: str, skill_name: str) -> str:
        """Extract skill instructions from SKILL.md"""
        # Remove frontmatter (YAML between --- ---)
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        
        # Clean up excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()
        
        return f"## SKILL: {skill_name}\n\n{content}"
    
    def _read_skill_doc(self, skill_path: Path, skill_name: str) -> Optional[str]:
        """Read and parse SKILL.md without publishing it yet."""
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            return None
        try:
            with open(skill_md, 'r', encoding='utf-8') as f:
                return self._parse_skill_md(
                    expand_pending_confirmation_copy(f.read()),
                    skill_name,
                )
        except OSError as e:
            logger.error(f"Failed to load SKILL.md for {skill_name}: {e}")
            return None

    def _commit_skill_doc(self, skill_name: str, parsed_doc: Optional[str]) -> None:
        """Publish a skill's documentation once its tools are known to work."""
        if parsed_doc is None:
            return
        self.skill_docs[skill_name] = parsed_doc
        logger.info(f"📚 Loaded documentation for skill: {skill_name}")

    @staticmethod
    def _tool_schema_name(tool: object) -> Optional[str]:
        if not isinstance(tool, dict):
            return None
        function = tool.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        return name if isinstance(name, str) and name else None

    def _register_module_tools(self, skill_name: str, module) -> None:
        """Merge a skill's TOOLS/TOOL_FUNCTIONS, skipping duplicate tool names."""
        schemas = getattr(module, 'TOOLS', None)
        functions = getattr(module, 'TOOL_FUNCTIONS', None)

        registered_schemas = 0
        if isinstance(schemas, list):
            for tool in schemas:
                name = self._tool_schema_name(tool)
                if not name:
                    logger.error(f"Skill {skill_name} declares a tool without a name, skipping it")
                    continue
                owner = self.tool_owners.get(name)
                if owner is not None:
                    logger.error(
                        f"Duplicate tool name '{name}': already registered by skill "
                        f"'{owner}', skipping the one from skill '{skill_name}'"
                    )
                    continue
                self.tool_owners[name] = skill_name
                self.tools.append(tool)
                self.tool_schemas[name] = tool
                registered_schemas += 1
            logger.info(f"🔧 Loaded {registered_schemas} tools from skill: {skill_name}")

        registered_functions = 0
        if isinstance(functions, dict):
            for name, func in functions.items():
                owner = self.tool_owners.get(name)
                if owner is not None and owner != skill_name:
                    logger.error(
                        f"Duplicate tool function '{name}': already registered by skill "
                        f"'{owner}', skipping the one from skill '{skill_name}'"
                    )
                    continue
                self.tool_owners[name] = skill_name
                self.tool_functions[name] = func
                registered_functions += 1
            logger.info(f"⚙️  Registered {registered_functions} functions from skill: {skill_name}")

    def load_skill(self, skill_path: Path):
        """Load a single skill (tools + documentation).

        The SKILL.md documentation is only published once ``tools.py`` has
        imported successfully. Otherwise a broken skill would still advertise
        tools that do not exist to the model.
        """
        skill_name = skill_path.name
        parsed_doc = self._read_skill_doc(skill_path, skill_name)

        tools_file = skill_path / "tools.py"
        if not tools_file.exists():
            logger.debug(f"Skill {skill_name} has no tools.py, skipping")
            self._commit_skill_doc(skill_name, parsed_doc)
            return

        try:
            spec = importlib.util.spec_from_file_location(
                f"skills.{skill_name}.tools",
                tools_file
            )
            if not (spec and spec.loader):
                logger.error(
                    f"Failed to create spec for skill {skill_name}; "
                    f"registering neither its tools nor its documentation"
                )
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(
                f"Failed to load skill {skill_name}: {e}; "
                f"registering neither its tools nor its documentation"
            )
            return

        self._register_module_tools(skill_name, module)
        self._commit_skill_doc(skill_name, parsed_doc)


    def get_tools(self) -> List[Dict]:
        """Get all loaded tools"""
        return self.tools
    
    def get_tool_function(self, name: str) -> Optional[Callable]:
        """Get a tool function by name"""
        return self.tool_functions.get(name)

    def get_tool_schema(self, name: str) -> Optional[Dict]:
        """Get a tool's OpenAI function-calling schema by name.

        Used by ToolExecutor to validate model-supplied arguments before
        dispatching them into the skill function.
        """
        return self.tool_schemas.get(name)
    
    def has_tools(self) -> bool:
        """Check if any tools are loaded"""
        return len(self.tools) > 0
    
    def get_skill_instructions(self) -> str:
        """Get combined instructions from all skills"""
        if not self.skill_docs:
            return ""
        
        instructions = "\n\n━━━━━━━━━━━━━━━━━━━━\n📚 功能说明文档\n━━━━━━━━━━━━━━━━━━━━\n\n"
        instructions += "\n\n".join(self.skill_docs.values())
        return instructions
