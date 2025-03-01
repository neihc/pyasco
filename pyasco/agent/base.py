import textwrap
from typing import List, Dict, Optional, Generator, Union, Any
from datetime import datetime
import re
import asyncio

from .types import Message
from .conversation import Conversation
from .memory_manager import MemoryManager
from .response_handler import ResponseHandler
from .tool_handler import ToolHandler
from .prompt import (
    DEFAULT_SYSTEM_PROMPT,
    FOLLOW_UP_PROMPT
)
from .utils import get_system_info

from ..logger_config import setup_logger
from ..config import Config
from ..services.llm import LLMService
from ..services.code_snippet_extractor import CodeSnippetExtractor
from ..services.lance_memory import LanceDBMemoryHandler
from ..services.embedding import EmbeddingService
from ..tools.code_execute import CodeExecutor


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
        return CodeExecutor()

    def _initialize_chat(self, context: str = "") -> None:
        system_info = get_system_info(self.python_executor)
        base_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{system_info}"
        system_content = f"{base_prompt}\n\n{self.custom_instructions}" if self.custom_instructions else base_prompt
        
        if context:
            system_content = f"{system_content}\n\nContext from your memory:\n{context}"
            
        self.logger.info(system_content)
        
        self.conversation.add_message(
            role="system",
            content=system_content
        )

    async def get_response(self, user_input: str, stream: bool = False) -> Union[Message, Generator[Message, None, None]]:
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


    async def ask(self, user_input: str, stream: bool = False, new_session: bool = False) -> Dict:
        """Process user input and get response"""
        self.logger.info(f"Getting response for user input (stream={stream}, new_session={new_session})")
        
        if new_session:
            # Get relevant context from memory
            context = ""
            if self.memory_manager:
                context = await self.memory_manager.get_context(user_input)
            
            # Reset conversation before starting new session
            self.conversation.clear()
            self._initialize_chat(context)
            
            content = user_input
        else:
            content = user_input
            
        if self.memory_manager:
            await self.memory_manager.remember(f"user: {user_input}")
            
        # Add message to conversation
        self.conversation.add_message(
            role="user",
            content=content
        )
        
        # Get response from LLM
        return self.response_handler.handle_response(
            self.conversation.to_llm_format(),
            self.model,
            self.conversation,
            stream
        )

    def get_follow_up(self, results: List[str]) -> str:
        output, _ = self.tool_handler.compress_results(results)
        base_prompt = FOLLOW_UP_PROMPT.format(output=output)
        
        # Check if agent is struggling (threshold of 10 exchanges)
        if len(self.conversation.messages) >= 6:
            struggle_suggestion = textwrap.dedent("""
                Note: I notice we've been going back and forth quite a bit
                To help resolve this more effectively, you could try remember old memories by using `search_context`

                ```python
                await search_context('<semantic query of memories you want to search>')
                ```
                """)
            return base_prompt + struggle_suggestion
        else:
            normal_suggestion = textwrap.dedent("""
                1. if there's error or don't include the neccessary information, response the next code to be run
                2. if done, based on output to answer user question at the first message

                Keep your answer short and directly. can use emoji if you need""")
            return base_prompt + normal_suggestion

    async def should_ask_user(self) -> bool:
        last_message = self.conversation.last_message
        if self.memory_manager:
            # Get the last assistant message if it exists
            if last_message and last_message.role == "assistant":
                await self.memory_manager.remember(f"assistant: {last_message.content}")
        
        
        return bool(last_message and last_message.tools)

    async def remember_conversation(self):
        """Trigger memory decay process"""
        if not self.memory_manager:
            self.logger.debug("Memory handling not enabled, skipping memory decay")
            return
            
        try:
            self.logger.info("Triggering memory decay process")
            await self.memory_manager.trigger_decay()
        except Exception as e:
            self.logger.error(f"Failed to trigger memory decay: {str(e)}")

    def reset(self):
        self.logger.info("Resetting agent state")
        self.conversation.clear()
        self.python_executor.reset()
        self._initialize_chat()
    
    async def stop_stream(self):
        """Stop the current streaming response"""
        self.logger.info("Stopping current stream")
        self.response_handler.stop_stream()

    def cleanup(self):
        self.logger.info("Cleaning up agent resources")
        self.python_executor.cleanup()

    def should_stop_follow_up(self, loop_count: int, max_loops: int = 5) -> bool:
        """Determine if we should stop the follow-up loop"""
        if loop_count >= max_loops:
            self.logger.warning(f"Reached maximum follow-up iterations ({max_loops})")
            return True
            
        last_message = self.conversation.last_message
        if not last_message:
            self.logger.debug("No last message found, stopping follow-up loop")
            return True
            
        if not last_message.tools:
            self.logger.debug("No tools found in last message, stopping follow-up loop")
            return True
            
        tool_count = len(last_message.tools)
        self.logger.debug(f"Found {tool_count} tool(s) in last message, continuing follow-up loop")
        return False

    def confirm(self) -> List[str] | None:
        """Execute any pending tools and return their results"""
        last_message = self.conversation.last_message
        if not last_message:
            self.logger.debug("No last message found, skipping tool execution")
            return None
            
        if not last_message.tools:
            self.logger.debug("No tools found in last message, skipping tool execution")
            return None
        
        tool_count = len(last_message.tools)
        self.logger.info(f"Executing {tool_count} tool(s) from last message")
        for i, tool in enumerate(last_message.tools):
            tool_type = tool.get('type', 'unknown')
            tool_name = tool.get('name', 'unnamed')
            tool_params = tool.get('parameters', {})
            
            # Format parameters for logging
            params_str = ', '.join([f"{k}={repr(v)}" for k, v in tool_params.items()])
            
            self.logger.info(f"Tool {i+1}/{tool_count}: {tool_type} - {tool_name}")
            self.logger.info(f"Parameters: {params_str}")
            
        return self.tool_handler.execute_tools(last_message.tools)
