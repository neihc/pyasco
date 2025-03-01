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
    context: Optional[Dict] = None

    def to_llm_format(self) -> Dict:
        """Convert message to format expected by LLM"""
        formatted_content = self.content
        if self.context:
            # Format context excluding embeddings and technical fields
            context_dict = {k: v for k, v in self.context.items() 
                          if k not in ('embeddings', 'id', 'node_id')}
            if context_dict:
                context_str = "\nContext:\n" + "\n".join(f"{k}: {v}" for k, v in context_dict.items())
                formatted_content = formatted_content + context_str

        return {
            "role": self.role,
            "content": formatted_content
        }

