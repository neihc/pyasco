from typing import Dict, Any, Optional, List
from ..services.llm import LLMService
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType

class MemoryManager:
    """Manages memory operations for the agent using LanceDB"""
    
    def __init__(self, memory_handler: LanceDBMemoryHandler):
        self.memory_handler = memory_handler

    def remember(self, content: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new short-term memory
        
        Args:
            content: The content to remember
            meta: Optional metadata about the memory
            
        Returns:
            str: ID of the created memory
        """
        memory_data = {
            'content': content,
            'memory_type': MemoryType.SHORT_TERM,
            'metadata': meta or {},
            'tags': []  # Could be enhanced to extract relevant tags
        }
        
        return self.memory_handler.add_memory(memory_data)

    def get_context(self, query: str, limit: int = 5) -> str:
        """
        Query relevant memories and combine them into context
        
        Args:
            query: The query to search memories with
            limit: Maximum number of memories to retrieve
            
        Returns:
            str: Combined context from relevant memories
        """
        # Search for relevant memories
        memories = self.memory_handler.search_similar(
            query=query,
            limit=limit,
            score_threshold=0.5  # Only include fairly relevant memories
        )
        
        if not memories:
            return ""
            
        # Combine memories into context string
        context_parts = []
        for memory in memories:
            # Add memory type and content
            context_parts.append(
                f"[{memory['memory_type']}] {memory['content']}"
            )
            
        return "\n\n".join(context_parts)
