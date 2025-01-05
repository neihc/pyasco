from typing import List, Dict, Optional, Generator, Union, Any
from datetime import datetime
import re

from ..logger_config import setup_logger
from .conversation import Conversation
from .memory_handler import MemoryHandler
from ..services.graphdb import GraphDB
from ..services.embedding import EmbeddingService
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
        self.graph_db = None
        self.embedding_service = None
        self.memory_handler = None
        
        if hasattr(config, 'memory') and config.memory.enabled:
            self.graph_db = GraphDB()
            if config.memory.graph_db:
                self.graph_db.configure(
                    uri=config.memory.graph_db.uri,
                    username=config.memory.graph_db.username,
                    password=config.memory.graph_db.password
                )
            
            self.embedding_service = EmbeddingService(
                #api_key=config.llm.api_key,
                #base_url=config.llm.base_url,
                model=config.embedding.model
            )
            
            self.memory_handler = MemoryHandler(
                memory_instructions=config.memory.instructions,
                llm_service=self.llm_service,
                graph_db=self.graph_db,
                embedding_service=self.embedding_service
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

    def get_response(self, user_input: str, stream: bool = False) -> Union[Message, Generator[Message, None, None]]:
        self.logger.info(f"Getting response for user input (stream={stream})")
        
        # Recall relevant context before handling the message
        if self.memory_handler:
            try:
                recalled_context = self.memory_handler.recall(user_input)
                if recalled_context:
                    context_summary = []
                    for result in recalled_context:
                        node = result.get('n')
                        if node and hasattr(node, 'get'):
                            # Skip if this content is already in the current conversation
                            content = node.get('original_content')
                            if content and not any(msg.content == content for msg in self.conversation.messages):
                                context_summary.append(f"Related context: {content}")
                    
                    if context_summary:
                        context_message = "\n\n".join(context_summary)
                        self.conversation.add_message(
                            role="system",
                            content=context_message
                        )
            except Exception as e:
                self.logger.error(f"Failed to recall context: {str(e)}")
        
        # Add message to history
        self.conversation.add_message(
            role="user",
            content=user_input
        )
        
        # Get LLM response through response handler
        return self.response_handler.handle_response(
            self.conversation.to_llm_format(), 
            self.model,
            self.conversation,
            stream
        )

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
                
            last_message = self.conversation.last_message
            results = self.tool_handler.execute_tools(last_message.tools if last_message else [])
            if not results:
                break
                
            follow_up = self.get_follow_up(results)
            current_response = self.get_response(follow_up, stream=stream)
            loop_count += 1
            
        return current_response

    def get_follow_up(self, results: List[str]) -> str:
        return FOLLOW_UP_PROMPT.format(output=chr(10).join(results))

    def should_ask_user(self) -> bool:
        last_message = self.conversation.last_message
        return bool(last_message and last_message.tools)

    def remember_conversation(self):
        """Store the current conversation in memory if memory handling is enabled"""
        if not self.memory_handler:
            self.logger.debug("Memory handling not enabled, skipping conversation storage")
            return
            
        try:
            # Convert conversation to storable format, excluding recalled context
            conversation_text = "\n".join([
                f"{msg.role}: {msg.content}" 
                for msg in self.conversation.messages 
                if msg.role != "system"  # Skip system messages
            ])
            
            if not conversation_text.strip():
                self.logger.debug("No conversation content to store")
                return
                
            # Check if this conversation is already stored
            if self.memory_handler and hasattr(self.memory_handler, 'graph_db'):
                existing = self.memory_handler.graph_db.execute_query(
                    "MATCH (n) WHERE n.conversation_id = $conv_id RETURN n",
                    {"conv_id": self.conversation_id}
                )
                if existing:
                    self.logger.debug(f"Conversation {self.conversation_id} already stored")
                    return
                
            # Add context about the conversation
            context = {
                "type": "conversation",
                "conversation_id": self.conversation_id,
                "timestamp": str(datetime.now()),
                "message_count": len(self.conversation.messages),
                **self.metadata
            }
            
            self.logger.info("Storing conversation in memory")
            self.memory_handler.remember(conversation_text, context)
            
        except Exception as e:
            self.logger.error(f"Failed to store conversation in memory: {str(e)}")

    def reset(self):
        self.logger.info("Resetting agent state")
        # Remember conversation before clearing it
        self.remember_conversation()
        self.conversation.clear()
        self.python_executor.reset()
        self._initialize_chat()
    
    def cleanup(self):
        self.logger.info("Cleaning up agent resources")
        # Remember conversation before cleanup
        self.remember_conversation()
        self.python_executor.cleanup()
        if self.graph_db:
            self.graph_db.close()


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
