from typing import List, Dict
from datetime import datetime
from .types import Message

class Conversation:
    """Manages the conversation history and message formatting"""
    
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str, user_id: str = "default", 
                   tools: List[Dict] = None) -> Message:
        """Add a new message to the conversation"""
        message = Message(
            role=role,
            content=content,
            user_id=user_id,
            tools=tools or [],
            timestamp=datetime.now()
        )
        self.messages.append(message)
        return message

    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation"""
        return self.messages

    def clear(self) -> None:
        """Clear all messages from the conversation"""
        self.messages = []

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
