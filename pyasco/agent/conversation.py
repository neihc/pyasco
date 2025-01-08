from typing import List, Dict
from datetime import datetime
from .types import Message

class Conversation:
    """Manages the conversation history and message formatting"""
    
    def __init__(self):
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str, user_id: str = "default", 
                   tools: List[Dict] = None, skills: List[Dict] = None,
                   context: Dict = None) -> Message:
        """Add a new message to the conversation"""
        message = Message(
            role=role,
            content=content,
            user_id=user_id,
            tools=tools or [],
            skills=skills or [],
            timestamp=datetime.now(),
            context=context
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
        formatted_messages = []
        for msg in self.messages:
            if msg.context and msg.context.get("type") == "memory_recall":
                recalled = msg.context.get("recalled", [])
                context_parts = ["Previous relevant context:"]
                
                for result in recalled:
                    if not isinstance(result, dict):
                        continue
                        
                    node = result.get('node', {})
                    score = result.get('score', 0.0)
                    
                    if score < 0.7:
                        continue
                    
                    properties = {k: v for k, v in dict(node).items() if k != 'embedding'}
                    labels = list(node.labels) if hasattr(node, 'labels') else ['Unknown']
                    label = labels[0] if labels else 'Unknown'
                    
                    content = properties.get('content', '')
                    if content:
                        context_parts.append(f"\n[{label}] (relevance: {score:.2f})")
                        context_parts.append(f"{content}")
                        
                        for key, value in properties.items():
                            if key not in ('content', 'embedding') and value:
                                context_parts.append(f"- {key}: {value}")
                
                if len(context_parts) > 1:
                    formatted_content = "\n".join(context_parts) + "\n\nCurrent message:\n" + msg.content
                    formatted_messages.append({
                        "role": msg.role,
                        "content": formatted_content
                    })
                else:
                    formatted_messages.append(msg.to_llm_format())
            else:
                formatted_messages.append(msg.to_llm_format())
                
        return formatted_messages

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
