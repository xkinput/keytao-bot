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
    
    def load_skill(self, skill_path: Path):
        """Load a single skill (tools + documentation)"""
        skill_name = skill_path.name
        
        # Load SKILL.md first
        skill_md = skill_path / "SKILL.md"
        if skill_md.exists():
            try:
                with open(skill_md, 'r', encoding='utf-8') as f:
                    content = f.read()
                    parsed_doc = self._parse_skill_md(content, skill_name)
                    self.skill_docs[skill_name] = parsed_doc
                    logger.info(f"📚 Loaded documentation for skill: {skill_name}")
            except Exception as e:
                logger.error(f"Failed to load SKILL.md for {skill_name}: {e}")
        
        # Load tools.py
        tools_file = skill_path / "tools.py"
        
        if not tools_file.exists():
            logger.debug(f"Skill {skill_path.name} has no tools.py, skipping")
            return
        
        try:
            # Load the tools module
            spec = importlib.util.spec_from_file_location(
                f"skills.{skill_path.name}.tools",
                tools_file
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                logger.error(f"Failed to create spec for skill {skill_path.name}")
                return
            
            # Get tools and functions
            if hasattr(module, 'TOOLS'):
                self.tools.extend(module.TOOLS)
                logger.info(f"🔧 Loaded {len(module.TOOLS)} tools from skill: {skill_name}")
            
            if hasattr(module, 'TOOL_FUNCTIONS'):
                self.tool_functions.update(module.TOOL_FUNCTIONS)
                logger.info(f"⚙️  Registered {len(module.TOOL_FUNCTIONS)} functions from skill: {skill_name}")
                
        except Exception as e:
            logger.error(f"Failed to load skill {skill_path.name}: {e}")
    
    def get_tools(self) -> List[Dict]:
        """Get all loaded tools"""
        return self.tools
    
    def get_tool_function(self, name: str) -> Optional[Callable]:
        """Get a tool function by name"""
        return self.tool_functions.get(name)
    
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
