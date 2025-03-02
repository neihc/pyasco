import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from .types import Message

class Conversation:
    """Manages the conversation history and message formatting"""
    
    def __init__(self):
        self.messages: List[Message] = []

    def __init__(self):
        self.messages: List[Message] = []
        self.memory_manager = None
        self._remember_tasks = []

    def set_memory_manager(self, memory_manager):
        """Set the memory manager for auto-remembering messages"""
        self.memory_manager = memory_manager

    def add_message(self, role: str, content: str, user_id: str = "default", 
                   tools: List[Dict] = None) -> Message:
        """Add a new message to the conversation and remember it in the background"""
        message = Message(
            role=role,
            content=content,
            user_id=user_id,
            tools=tools or [],
            timestamp=datetime.now()
        )
        self.messages.append(message)
        
        # Auto-remember in background if memory manager is set
        if self.memory_manager and role != "system":
            task = asyncio.create_task(self._remember_message(message))
            self._remember_tasks.append(task)
            # Clean up completed tasks
            self._remember_tasks = [t for t in self._remember_tasks if not t.done()]
            
        return message

    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation"""
        return self.messages

    async def clear(self) -> None:
        """Clear all messages from the conversation, waiting for any pending remember tasks"""
        # Wait for any pending remember tasks to complete
        if hasattr(self, '_remember_tasks'):
            for task in self._remember_tasks:
                if not task.done():
                    await task
        
        self.messages = []
        
        # Reset the remember tasks list
        self._remember_tasks = []

    def to_llm_format(self) -> List[Dict]:
        """Convert conversation to format expected by LLM"""
        return [msg.to_llm_format() for msg in self.messages]

    def to_text_format(self, skip_system: bool = True) -> str:
        """Convert conversation to plain text format
        
        Args:
            skip_system: Whether to skip the system message (default: True)
            
        Returns:
            String representation of the conversation with each message
            formatted as "{role}: {content}"
        """
        messages = self.messages[1:] if skip_system and self.messages else self.messages
        return "\n".join(f"{msg.role}: {msg.content}" for msg in messages)

    @property
    def last_message(self) -> Message:
        """Get the last message in the conversation"""
        return self.messages[-1] if self.messages else None
        
    async def _remember_message(self, message: Message) -> None:
        """Remember a message in the memory manager (internal method)"""
        if not self.memory_manager or not message:
            return
            
        # Format the message as "role: content" for the memory
        memory_content = f"{message.role}: {message.content}"
        await self.memory_manager.remember(memory_content)
