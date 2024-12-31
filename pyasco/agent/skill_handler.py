from typing import List, Optional
from ..services.skill_manager import SkillManager, Skill
from ..services.llm import get_openai_response
from ..tools.code_execute import CodeExecutor

class SkillHandler:
    def __init__(self, skill_manager: SkillManager, executor: CodeExecutor):
        self.skill_manager = skill_manager
        self.executor = executor

    def get_relevant_skills(self, user_input: str, model: str) -> List[Skill]:
        """Get relevant skills for the given user input"""
        available_skills = list(self.skill_manager.skills.keys())
        if not available_skills:
            return []

        skill_list = "\n".join(f"- {skill}" for skill in available_skills)
        skill_prompt = (
            "Based on the following user input, select which of these skills would be most relevant. "
            "Reply with ONLY the exact skill names, one per line, maximum 3 skills. "
            "If no skills are relevant, reply with 'none'.\n\n"
            f"User input: {user_input}\n\n"
            f"Available skills:\n{skill_list}\n\n"
            "Selected skills:"
        )

        skill_response = get_openai_response([{
            "role": "user",
            "content": skill_prompt
        }], model=model)

        selected_skill_names = [name.strip() for name in skill_response.split('\n') if name.strip()]
        relevant_skills = []
        
        for name in selected_skill_names:
            if name.lower() != 'none' and name in self.skill_manager.skills:
                relevant_skills.append(self.skill_manager.skills[name])

        return relevant_skills

    def process_skills(self, user_input: str, relevant_skills: List[Skill], messages: List[dict]) -> str:
        """Process skills and update user input with skill information"""
        if not relevant_skills:
            return user_input

        # Check for existing skills in previous messages
        existing_skills = set()
        for msg in messages:
            if msg.get("skills"):
                existing_skills.update(skill["name"] for skill in msg["skills"])

        # Only process new skills
        new_skills = [skill for skill in relevant_skills if skill.name not in existing_skills]
        if not new_skills:
            return user_input

        # Add skill information to user input
        skills_info = "\n\nLoading and making available these relevant skills:\n\n"
        for skill in new_skills:
            skill_code = self.skill_manager.get_skill_code(skill)

            # Install requirements if any
            if skill.requirements:
                req_install = f"pip install {' '.join(skill.requirements)}"
                stdout, stderr = self.executor.execute(req_install, 'bash')
                if stderr and "ERROR:" in stderr:
                    continue

            # Execute the skill code
            stdout, stderr = self.executor.execute(skill_code)

            skills_info += f"### {skill.name}\n"
            skills_info += f"**Usage:** {skill.usage}\n"
            skills_info += f"**Note:** This skill's functions are now loaded and ready to use. "
            skills_info += f"Do not redefine them unless you need to modify their behavior.\n"
            skills_info += f"**Code:**\n```python\n{skill_code}\n```\n\n"

        return user_input + skills_info
