from typing import List, Dict
from datetime import datetime
from .types import Message

class Conversation:
    """Manages the conversation history and message formatting"""
    
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str, user_id: str = "default", 
                   tools: List[Dict] = None, skills: List[Dict] = None) -> Message:
        """Add a new message to the conversation"""
        message = Message(
            role=role,
            content=content,
            user_id=user_id,
            tools=tools or [],
            skills=skills or [],
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

    @property
    def last_message(self) -> Message:
        """Get the last message in the conversation"""
        return self.messages[-1] if self.messages else None
