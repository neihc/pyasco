from typing import List, Dict, Optional, Generator, Union, Any
import re

from ..logger_config import setup_logger
from ..services.skill_manager import Skill
from .prompt import DEFAULT_SYSTEM_PROMPT, FOLLOW_UP_PROMPT
from .types import AgentResponse
from ..config import Config
from ..services.llm import configure_client
from ..services.code_snippet_extractor import CodeSnippetExtractor
from ..services.skill_manager import SkillManager
from ..tools.code_execute import CodeExecutor
from .skill_handler import SkillHandler
from .response_handler import ResponseHandler
from .tool_handler import ToolHandler
from .utils import get_system_info


class Agent:
    def __init__(self, config: Config):
        self.logger = setup_logger('agent')
        self.logger.info("Initializing Agent")
        self.messages: List[Dict] = []
        self.code_extractor = CodeSnippetExtractor()
        self.python_executor = self._setup_executor(config)
        self.custom_instructions = config.custom_instructions or ""
        self.model = config.llm.model
        
        configure_client(api_key=config.llm.api_key, base_url=config.llm.base_url)
        self.skill_manager = SkillManager(config.skills_path)
        
        # Initialize handlers
        self.skill_handler = SkillHandler(self.skill_manager, self.python_executor)
        self.response_handler = ResponseHandler(self.code_extractor)
        self.tool_handler = ToolHandler(self.python_executor)
        
        self._initialize_chat()

    def _setup_executor(self, config: Config) -> CodeExecutor:
        docker_options = None
        if config.docker.use_docker:
            docker_options = {
                'mem_limit': config.docker.mem_limit,
                'cpu_count': config.docker.cpu_count,
                'volumes': {}
            }
            
            if config.docker.volumes:
                docker_options['volumes'].update(config.docker.volumes)
            
            if config.skills_path not in docker_options['volumes']:
                docker_options['volumes'][config.skills_path] = {
                    'bind': '/skills',
                    'mode': 'ro'
                }
        
        return CodeExecutor(
            use_docker=config.docker.use_docker,
            docker_image=config.docker.image,
            docker_options=docker_options,
            bash_shell=config.docker.bash_command,
            python_command=config.docker.python_command,
            env_file=config.docker.env_file
        )

    def _initialize_chat(self) -> None:
        system_info = get_system_info(self.python_executor.use_docker,
                                    self.python_executor.docker_image)
        base_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{system_info}"
        system_content = f"{base_prompt}\n\n{self.custom_instructions}" if self.custom_instructions else base_prompt
        self.messages.append({
            "role": "system",
            "content": system_content
        })

    def get_response(self, user_input: str, stream: bool = False) -> Union[AgentResponse, Generator[AgentResponse, None, None]]:
        self.logger.info(f"Getting response for user input (stream={stream})")
        
        # Get relevant skills using skill handler
        relevant_skills = self.skill_handler.get_relevant_skills(user_input, self.model)
        
        # Process skills and update input
        user_input = self.skill_handler.process_skills(user_input, relevant_skills, self.messages)
        
        # Add message to history
        self.messages.append({
            "role": "user", 
            "content": user_input,
            "skills": [skill.to_dict() for skill in relevant_skills] if relevant_skills else []
        })
        
        # Get LLM response through response handler
        return self.response_handler.handle_response(self.messages, self.model, stream)

    def ask(self, new_input: str, stream: bool = False, auto: bool = False, max_loops: int = 5) -> Dict:
        response = self.get_response(new_input, stream=stream)
        
        if not auto:
            return response
            
        loop_count = 0
        current_response = response
        
        while True:
            if not self.should_ask_user():
                break
                
            if loop_count >= max_loops:
                self.logger.warning(f"Reached maximum follow-up iterations ({max_loops})")
                break
                
            results = self.tool_handler.execute_tools(self.messages[-1].get("tools", []))
            if not results:
                break
                
            follow_up = self.get_follow_up(results)
            current_response = self.get_response(follow_up, stream=stream)
            loop_count += 1
            
        return current_response

    def get_follow_up(self, results: List[str]) -> str:
        return FOLLOW_UP_PROMPT.format(output=chr(10).join(results))

    def should_ask_user(self) -> bool:
        if not self.messages:
            return False
        last_message = self.messages[-1]
        return bool(last_message.get("tools"))

    def reset(self):
        self.logger.info("Resetting agent state")
        self.messages = []
        self.python_executor.reset()
        self._initialize_chat()
    
    def cleanup(self):
        self.logger.info("Cleaning up agent resources")
        self.python_executor.cleanup()

    def parse_skill_response(self, response_content: str) -> Dict[str, Any]:
        """Parse a skill learning response into components.
        
        Args:
            response_content: The LLM response content
            
        Returns:
            Dict containing name, usage, code, and requirements
            
        Raises:
            ValueError: If response format is invalid
        """
        try:
            # Extract skill name
            name_match = re.search(r"SKILL NAME:\s*(.+?)(?=\n|$)", response_content)
            name = name_match.group(1).strip() if name_match else None

            # Extract usage
            usage_match = re.search(r"USAGE:\s*(.+?)(?=\n|$)", response_content)
            usage = usage_match.group(1).strip() if usage_match else None

            # Extract requirements
            req_match = re.search(r"REQUIREMENTS:\s*(.+?)(?=\n|$)", response_content)
            requirements_str = req_match.group(1).strip() if req_match else "none"
            requirements = [r.strip() for r in requirements_str.split(",")] if requirements_str.lower() != "none" else []

            # Extract code
            code_snippets = self.code_extractor.extract_snippets(response_content)
            code = code_snippets[0].content if code_snippets else None

            if not all([name, usage, code]):
                raise ValueError("Missing required skill components")

            return {
                "name": name,
                "usage": usage,
                "file_path": f"{name.lower().replace(' ', '_')}.py",
                "requirements": requirements,
                "code": code
            }
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Invalid skill response format: {str(e)}")

    def learn_that_skill(self) -> 'Skill':
        """Convert the current conversation into a reusable skill.
        Takes the conversation history and asks the LLM to consolidate it
        into a skill in the correct format.
        
        Returns:
            Skill: The newly learned skill
            
        Raises:
            ValueError: If skill response is invalid or no messages to learn from
        """
        if len(self.messages) < 2:  # Need at least system + 1 interaction
            raise ValueError("Not enough conversation history to learn from")
            
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Create prompt to consolidate conversation into skill
                conversation = "\n".join(f"{msg['role']}: {msg['content']}" 
                                    for msg in self.messages[1:])  # Skip system message
                
                error_feedback = ""
                if last_error:
                    error_feedback = (
                        f"\n\nPrevious attempt failed with error: {last_error}"
                        "\nPlease ensure your response includes ALL required sections:"
                        "\n- SKILL NAME (required)"
                        "\n- USAGE (required)"
                        "\n- REQUIREMENTS (required, use 'none' if no requirements)"
                        "\n- CODE section with ```python code block (required)"
                    )
                
                prompt = (
                    "Based on our conversation above, please consolidate the key functionality "
                    "into a reusable skill. You MUST include ALL of these sections in your response:"
                    "\n\nSKILL NAME: <name of the skill>"
                    "\nUSAGE: <brief description of what the skill does and how to use it>"
                    "\nREQUIREMENTS: <comma-separated list of pip packages, or 'none' if no requirements>"
                    "\nCODE:\n"
                    "```python\n"
                    "<the actual Python code for the skill>\n"
                    "```"
                    f"{error_feedback}"
                )
        
                response = self.get_response(prompt)
                if isinstance(response, Generator):
                    # Handle streaming response
                    content = ""
                    for chunk in response:
                        content += chunk.content
                    response_content = content
                else:
                    response_content = response.content
                    
                skill_data = self.parse_skill_response(response_content)
                
                return self.skill_handler.skill_manager.learn(
                    name=skill_data["name"],
                    usage=skill_data["usage"],
                    code=skill_data["code"],
                    requirements=skill_data["requirements"]
                )
                
            except ValueError as e:
                last_error = str(e)
                retry_count += 1
                self.logger.warning(f"Attempt {retry_count} failed: {last_error}")
                
                if retry_count >= max_retries:
                    raise ValueError(
                        f"Failed to learn skill after {max_retries} attempts. "
                        f"Last error: {last_error}"
                    )
                
                continue

    def improve_that_skill(self, skill_name: Optional[str] = None) -> Optional['Skill']:
        """Improve an existing skill based on the current conversation.
        Takes the conversation history and asks the LLM to improve the specified skill.
        If no skill_name is provided, automatically determines which skill to improve.
        
        Args:
            skill_name: Optional name of the skill to improve. If None, auto-determines from conversation.
            
        Returns:
            Skill: The improved skill or None if skill not found
            
        Raises:
            ValueError: If skill response is invalid or no messages to learn from
        """
        if len(self.messages) < 2:
            raise ValueError("Not enough conversation history to learn from")
            
        error_context = []
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if skill_name is None:
                    # Create prompt to identify skill from conversation
                    conversation = "\n".join(f"{msg['role']}: {msg['content']}" 
                                        for msg in self.messages[1:])
                    
                    identify_prompt = (
                        f"{conversation}\n"
                        "Based on our conversation above, which of these existing skills would be most "
                        "appropriate to improve? Reply with JUST the skill name, nothing else.\n\n"
                        "Available skills:\n"
                    )
                    
                    # Add available skills to prompt
                    for name, skill in self.skill_handler.skill_manager.skills.items():
                        identify_prompt += f"- {name}: {skill.usage}\n"
                    
                    response = self.get_response(identify_prompt)
                    if isinstance(response, Generator):
                        content = ""
                        for chunk in response:
                            content += chunk.content
                        skill_name = content.strip()
                    else:
                        skill_name = response.content.strip()
                    
                # Get existing skill
                skill = self.skill_handler.skill_manager.get_skill(skill_name)
                if not skill:
                    raise ValueError(f"Skill '{skill_name}' not found")
                    
                # Get existing code
                existing_code = self.skill_handler.skill_manager.get_skill_code(skill)
                    
                # Create prompt to improve skill
                conversation = "\n".join(f"{msg['role']}: {msg['content']}" 
                                    for msg in self.messages[1:])
                
                # Add error context if any previous attempts failed
                error_info = ""
                if error_context:
                    error_info = "\nPrevious attempts failed with these errors:\n" + "\n".join(
                        f"Attempt {i+1}: {err}" for i, err in enumerate(error_context)
                    ) + "\nPlease address these issues in your improvement."
                
                prompt = (
                    f"{conversation}\n"
                    f"Based on our conversation above and the existing skill below, please improve "
                    f"the skill by updating its usage description and code as needed. You may need to put more information on usage to avoid mistake this time{error_info}\n\n"
                    f"EXISTING SKILL:\n"
                    f"NAME: {skill.name}\n"
                    f"USAGE: {skill.usage}\n"
                    f"Please provide the improved version in this format, Do not change SKILL NAME:\n"
                    f"SKILL NAME: {skill.name}\n"
                    f"USAGE: <improved description of what the skill does and how to use it>\n"
                    f"REQUIREMENTS: <comma-separated list of pip packages, or 'none' if no requirements>\n"
                    f"CODE:\n"
                    f"```python\n"
                    f"<improved Python code for the skill>\n"
                    f"```\n"
                )
                
                response = self.get_response(prompt)
                if isinstance(response, Generator):
                    content = ""
                    for chunk in response:
                        content += chunk.content
                    response_content = content
                else:
                    response_content = response.content
                    
                skill_data = self.parse_skill_response(response_content)
                
                # Verify the skill name matches
                if skill_data["name"] != skill_name:
                    raise ValueError(f"Skill name mismatch: expected {skill_name}, got {skill_data['name']}")
                
                return self.skill_handler.skill_manager.improve_skill(
                    name=skill_data["name"],
                    usage=skill_data["usage"],
                    code=skill_data["code"],
                    requirements=skill_data["requirements"]
                )
                
            except Exception as e:
                self.logger.error(f"Attempt {retry_count + 1} failed: {str(e)}")
                error_context.append(str(e))
                retry_count += 1
                
                if retry_count >= max_retries:
                    self.logger.error(f"Failed to improve skill after {max_retries} attempts")
                    raise ValueError(f"Failed to improve skill after {max_retries} attempts. Errors: {error_context}")

    def should_stop_follow_up(self, loop_count: int, max_loops: int = 5) -> bool:
        """Determine if we should stop the follow-up loop"""
        if loop_count >= max_loops:
            self.logger.warning(f"Reached maximum follow-up iterations ({max_loops})")
            return True
            
        if not self.messages:
            return True
            
        last_message = self.messages[-1]
        if not last_message.get("tools"):
            return True
            
        return False

    def confirm(self) -> List[str] | None:
        """Execute any pending tools and return their results"""
        if not self.messages:
            return None
            
        last_message = self.messages[-1]
        if not last_message.get("tools"):
            return None
            
        return self.tool_handler.execute_tools(last_message["tools"])
