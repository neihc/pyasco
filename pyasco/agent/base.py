from typing import List, Dict, Optional, Generator, Union, Any
from datetime import datetime
import re
import asyncio

from ..logger_config import setup_logger
from .conversation import Conversation
from ..services.lance_memory import LanceDBMemoryHandler
from ..services.embedding import EmbeddingService
from .memory_manager import MemoryManager
from .prompt import (
    DEFAULT_SYSTEM_PROMPT,
    FOLLOW_UP_PROMPT
)
from .types import Message
from ..config import Config
from ..services.llm import LLMService
from ..services.code_snippet_extractor import CodeSnippetExtractor
from ..services.skill_manager import SkillManager
from ..tools.code_execute import CodeExecutor
from .response_handler import ResponseHandler
from .tool_handler import ToolHandler
from .utils import get_system_info


class Agent:
    def __init__(self, config: Config, user_id: str = "0", app_type: str = "console", **metadata):
        self.logger = setup_logger('agent')
        self.logger.info("Initializing Agent")
        self.user_id = user_id
        self.app_type = app_type
        self.conversation_id = str(int(datetime.now().timestamp()))
        self.metadata = {
            "user_id": user_id,
            "app_type": app_type,
            "conversation_id": self.conversation_id,
            **metadata
        }
        self.conversation = Conversation()
        self.code_extractor = CodeSnippetExtractor()
        self.python_executor = self._setup_executor(config)
        self.custom_instructions = config.custom_instructions or ""
        self.model = config.llm.model
        
        self.llm_service = LLMService(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=self.model
        )
        
        # Initialize memory services if configured
        self.embedding_service = None
        self.memory_handler = None
        self.memory_manager = None
        
        if hasattr(config, 'memory') and config.memory.enabled:
            self.embedding_service = EmbeddingService()
            
            self.memory_handler = LanceDBMemoryHandler(
                db_path=config.memory.db_path if hasattr(config.memory, 'db_path') else "~/.pyasco/memories"
            )
            
            self.memory_manager = MemoryManager(
                memory_handler=self.memory_handler,
                llm_service=self.llm_service
            )
        
        # Initialize handlers
        self.response_handler = ResponseHandler(self.code_extractor, self.llm_service)
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
        system_info = get_system_info(self.python_executor)
        base_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{system_info}"
        system_content = f"{base_prompt}\n\n{self.custom_instructions}" if self.custom_instructions else base_prompt
        self.logger.info(system_content)
        
        self.conversation.add_message(
            role="system",
            content=system_content
        )



    async def _get_response_with_recall(self, user_input: str, stream: bool = False) -> Union[Message, Generator[Message, None, None]]:
        """Get response with memory recall"""
        self.logger.info(f"Getting response for user input with recall (stream={stream})")
        
        # Reset conversation before starting
        self.conversation.clear()
        self._initialize_chat()
        
        # Get relevant context from memory
        context = ""
        if self.memory_manager:
            context = await self.memory_manager.get_context(user_input)
            
        # Combine context with user input if we have context
        content = user_input
        if context:
            content = f"Context from previous conversations:\n{context}\n\nCurrent input:\n{user_input}"
            
        # Add combined message to conversation
        self.conversation.add_message(
            role="user",
            content=content
        )
        
        # Store user input in memory
        if self.memory_manager:
            await self.memory_manager.remember(f"user: {user_input}")
        
        # Get response from LLM
        response = self.response_handler.handle_response(
            self.conversation.to_llm_format(),
            self.model,
            self.conversation,
            stream
        )
        
        return response

    def get_response(self, user_input: str, stream: bool = False) -> Union[Message, Generator[Message, None, None]]:
        """Get response without recall for follow-up messages"""
        self.logger.info(f"Getting response for user input (stream={stream})")
        
        self.conversation.add_message(
            role="user",
            content=user_input
        )
        
        return self.response_handler.handle_response(
            self.conversation.to_llm_format(),
            self.model,
            self.conversation,
            stream
        )

    async def ask(self, user_input: str, stream: bool = False, auto: bool = False, recall: bool = False, max_loops: int = 5) -> Dict:
        """Process user input and handle any follow-up interactions"""
        # Use recall if explicitly requested or in auto mode
        if recall or auto:
            response = await self._get_response_with_recall(user_input, stream=stream)
        else:
            response = self.get_response(user_input, stream=stream)
        
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
                
            last_message = self.conversation.last_message
            if self.memory_manager and last_message and last_message.role == "assistant":
                await self.memory_manager.remember(f"assistant: {last_message.content}")

            results = self.tool_handler.execute_tools(last_message.tools if last_message else [])
            if not results:
                break
                
            follow_up = self.get_follow_up(results)
            # Use regular get_response for follow-ups (no recall)
            current_response = self.get_response(follow_up, stream=stream)
            if auto:
                # Execute any tools from the follow-up response
                if self.should_ask_user():
                    results = self.confirm()
            loop_count += 1
            
        return current_response

    def get_follow_up(self, results: List[str]) -> str:
        return FOLLOW_UP_PROMPT.format(output=chr(10).join(results))

    def should_ask_user(self) -> bool:
        last_message = self.conversation.last_message
        return bool(last_message and last_message.tools)

    def remember_conversation(self):
        """Trigger memory decay process"""
        if not self.memory_manager:
            self.logger.debug("Memory handling not enabled, skipping memory decay")
            return
            
        try:
            self.logger.info("Triggering memory decay process")
            asyncio.run(self.memory_manager.trigger_decay())
        except Exception as e:
            self.logger.error(f"Failed to trigger memory decay: {str(e)}")

    def reset(self):
        self.logger.info("Resetting agent state")
        self.conversation.clear()
        self.python_executor.reset()
        self._initialize_chat()
    
    def cleanup(self):
        self.logger.info("Cleaning up agent resources")
        self.python_executor.cleanup()
        if self.memory_handler:
            del self.memory_handler


    def should_stop_follow_up(self, loop_count: int, max_loops: int = 5) -> bool:
        """Determine if we should stop the follow-up loop"""
        if loop_count >= max_loops:
            self.logger.warning(f"Reached maximum follow-up iterations ({max_loops})")
            return True
            
        last_message = self.conversation.last_message
        if not last_message:
            return True
            
        if not last_message.tools:
            return True
            
        return False

    def confirm(self) -> List[str] | None:
        """Execute any pending tools and return their results"""
        last_message = self.conversation.last_message
        if not last_message:
            return None
            
        if not last_message.tools:
            return None
            
        return self.tool_handler.execute_tools(last_message.tools)
