from dataclasses import dataclass
from typing import List, Dict

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
