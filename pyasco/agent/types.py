from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import uuid

@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: str
    content: str
    user_id: str = "default"
    tools: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    skills: List[Dict] = field(default_factory=list)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: Optional[str] = None

    def to_llm_format(self) -> Dict:
        """Convert message to format expected by LLM"""
        return {
            "role": self.role,
            "content": self.content
        }

