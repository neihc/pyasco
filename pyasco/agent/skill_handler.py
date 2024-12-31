from typing import List, Optional
from ..services.skill_manager import SkillManager, Skill
from ..services.llm import get_openai_response
from ..tools.code_execute import CodeExecutor

class SkillHandler:
    def __init__(self, skill_manager: SkillManager, executor: CodeExecutor):
        self.skill_manager = skill_manager
        self.executor = executor

    def get_relevant_skills(self, messages: List[dict], model: str) -> List[Skill]:
        """Get relevant skills based on conversation history"""
        available_skills = list(self.skill_manager.skills.keys())
        if not available_skills:
            return []

        # Get last few messages for context, excluding system message
        user_msgs = [msg for msg in messages if msg['role'] != 'system']
        recent_msgs = user_msgs[-3:] if len(user_msgs) > 3 else user_msgs
        conversation = "\n".join(f"{msg['role']}: {msg['content']}" for msg in recent_msgs)

        skill_list = "\n".join(f"{skill}: {self.skill_manager.skills[skill].usage}" 
                              for skill in available_skills)
        
        skill_prompt = (
            f"Available Skills:\n{skill_list}\n\n"
            f"Recent Conversation:\n{conversation}\n\n"
            "Based on the conversation, list up to 3 most relevant skills that would be helpful.\n"
            "Format your response as a numbered list with one skill name per line:\n"
            "1. skill_name\n"
            "2. skill_name\n"
            "3. skill_name\n\n"
            "If no skills are relevant, just respond with: none"
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
            skills_info += f"**Code:**\n```python\n{skill_code}\n```\n\n"
            skills_info += f"**Note:** This code has been executed in current notebook with this output:\n{stdout}\n\n{stderr}\n\n"
            skills_info += f"Do not redefine them unless you need to modify their behavior.\n"

        return user_input + skills_info
