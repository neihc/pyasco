from typing import List, Dict, Optional, Generator, Union

from ..logger_config import setup_logger
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
        base_prompt = f"{self.DEFAULT_SYSTEM_PROMPT}\n\n{system_info}"
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
