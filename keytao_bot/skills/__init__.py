"""
Skills loader for AI chat
加载和管理 AI skills
"""
import os
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
        
    def load_all_skills(self):
        """Load all skills from skills directory"""
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return
        
        for skill_path in self.skills_dir.iterdir():
            if skill_path.is_dir() and not skill_path.name.startswith('.'):
                self.load_skill(skill_path)
    
    def load_skill(self, skill_path: Path):
        """Load a single skill"""
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
                logger.info(f"Loaded {len(module.TOOLS)} tools from skill: {skill_path.name}")
            
            if hasattr(module, 'TOOL_FUNCTIONS'):
                self.tool_functions.update(module.TOOL_FUNCTIONS)
                logger.info(f"Registered {len(module.TOOL_FUNCTIONS)} functions from skill: {skill_path.name}")
                
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
