from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: str
    content: str
    user_id: str = "default"
    tools: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    skills: List[Dict] = field(default_factory=list)

    def to_llm_format(self) -> Dict:
        """Convert message to format expected by LLM"""
        return {
            "role": self.role,
            "content": self.content
        }

@dataclass
class AgentResponse:
    """Represents a response from the agent"""
    role: str = "assistant"
    content: str = ""
    tools: List[Dict] = None
    done: bool = True

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
