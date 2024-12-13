from dataclasses import dataclass
from typing import List, Optional, Dict, Set
import os
import json
from pathlib import Path
import subprocess
import sys

@dataclass
class Skill:
    name: str
    usage: str
    file_path: str
    requirements: List[str] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "usage": self.usage,
            "file_path": self.file_path,
            "requirements": self.requirements or []
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Skill':
        return cls(**data)

class SkillManager:
    def __init__(self, skills_path: str):
        self.skills_path = Path(skills_path)
        self.skills: Dict[str, Skill] = {}
        self._ensure_skills_structure()
        self._load_skills()
    
    def _ensure_skills_structure(self):
        """Ensure skills directory structure exists"""
        self.skills_path.mkdir(exist_ok=True)
        
        # Create requirements.txt if it doesn't exist
        req_file = self.skills_path / "requirements.txt"
        if not req_file.exists():
            req_file.touch()
    
    def _load_skills(self):
        """Load all skills from the skills directory"""
        manifest_path = self.skills_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                for skill_data in manifest['skills']:
                    # Filter out any fields that aren't part of the Skill class
                    filtered_data = {
                        "name": skill_data["name"],
                        "usage": skill_data["usage"],
                        "file_path": skill_data["file_path"],
                        "requirements": skill_data.get("requirements", [])
                    }
                    skill = Skill.from_dict(filtered_data)
                    self.skills[skill.name] = skill
    
    def _save_manifest(self):
        """Save skills manifest"""
        manifest = {
            'skills': [skill.to_dict() for skill in self.skills.values()]
        }
        with open(self.skills_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _update_requirements(self, new_requirements: List[str]):
        """Update requirements.txt with new requirements"""
        req_file = self.skills_path / "requirements.txt"
        existing_requirements: Set[str] = set()
        
        # Read existing requirements
        if req_file.exists():
            with open(req_file, 'r') as f:
                existing_requirements = {line.strip() for line in f if line.strip()}
        
        # Add new requirements
        updated_requirements = existing_requirements.union(new_requirements)
        
        # Write back all requirements
        with open(req_file, 'w') as f:
            for req in sorted(updated_requirements):
                f.write(f"{req}\n")
    
    def _install_requirements(self):
        """Install requirements using pip"""
        req_file = self.skills_path / "requirements.txt"
        if not req_file.exists() or req_file.stat().st_size == 0:
            return
            
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", str(req_file)
            ])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install requirements: {e}")
    
    def learn(self, name: str, usage: str, code: str, requirements: List[str] = None) -> Skill:
        """Add a new skill"""
        file_path = f"{name.lower().replace(' ', '_')}.py"
        full_path = self.skills_path / file_path
        
        # Save the skill code
        with open(full_path, 'w') as f:
            f.write(code)
        
        # Update requirements if provided
        if requirements:
            self._update_requirements(requirements)
            
        # Create and save skill metadata
        skill = Skill(
            name=name, 
            usage=usage, 
            file_path=str(file_path),
            requirements=requirements
        )
        self.skills[name] = skill
        self._save_manifest()
        return skill
    
    def get_all_skills(self) -> List[Skill]:
        """Get all available skills"""
        return list(self.skills.values())
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a specific skill by name"""
        return self.skills.get(name)
    
    def get_skills_path(self) -> str:
        """Get the absolute path to skills directory"""
        return str(self.skills_path.absolute())
    
    def get_requirements(self) -> List[str]:
        """Get all requirements from requirements.txt"""
        req_file = self.skills_path / "requirements.txt"
        if not req_file.exists():
            return []
            
        with open(req_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
            
    def get_skill_code(self, skill: Skill) -> str:
        """Read and return the code content for a skill
        
        Args:
            skill: The skill to get code for
            
        Returns:
            str: The skill's code content or empty string if file not found
        """
        try:
            with open(self.skills_path / skill.file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ""
            
    def improve_skill(self, name: str, usage: str, code: str, requirements: List[str] = None) -> Optional[Skill]:
        """Improve an existing skill with updated content
        
        Args:
            name: Name of the skill to improve
            usage: Updated usage description
            code: Updated code content
            requirements: Updated requirements list
            
        Returns:
            Updated Skill object or None if skill not found
        """
        if name not in self.skills:
            return None
            
        # Get existing skill
        skill = self.skills[name]
        
        # Update the code file
        with open(self.skills_path / skill.file_path, 'w') as f:
            f.write(code)
            
        # Update requirements if provided
        if requirements:
            self._update_requirements(requirements)
            
        # Update skill metadata
        skill.usage = usage
        skill.requirements = requirements
        
        # Save updated manifest
        self._save_manifest()
        
        return skill
