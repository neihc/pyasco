from typing import Dict, Any, Optional, List
import math
from datetime import datetime
from collections import defaultdict
import asyncio
from ..services.llm import LLMService
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType

class MemoryManager:
    """Manages memory operations for the agent using LanceDB"""
    
    def __init__(self, memory_handler: LanceDBMemoryHandler, token_window: int = 2000):
        self.memory_handler = memory_handler
        self.token_window = token_window

    def _calculate_decay_score(self, memory: Dict[str, Any]) -> float:
        """Calculate time-based decay score for a memory"""
        now = datetime.now()
        age = (now - memory['created_at']).total_seconds()
        # Decay factor: newer memories get higher scores
        return math.exp(-age / (7 * 24 * 3600))  # 7 days half-life

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count"""
        return len(text.split()) * 1.3  # Rough approximation

    def _format_memories_by_type(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories grouped by type"""
        grouped = defaultdict(list)
        for memory in memories:
            grouped[memory['memory_type']].append(memory['content'])

        sections = []
        for memory_type in [MemoryType.SHORT_TERM, MemoryType.LONG_TERM, MemoryType.REFLECTION]:
            if memory_type in grouped:
                sections.append(f"{memory_type.upper()}:")
                sections.extend(grouped[memory_type])
                sections.append("")  # Empty line between sections

        return "\n".join(sections).strip()

    async def remember(self, content: str, meta: Optional[Dict[str, Any]] = None) -> str:
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
        
        return await self.memory_handler.add_memory(memory_data)

    async def get_context(self, query: str) -> str:
        """
        Get relevant context from different types of memories
        
        Args:
            query: The query to search memories with
            
        Returns:
            str: Formatted context from relevant memories
        """
        # Fetch different types of memories in parallel
        async def get_short_term():
            return await self.memory_handler.search_similar(
                query="",  # Empty query to get latest
                limit=10,
                filter_dict={"memory_type": MemoryType.SHORT_TERM}
            )

        async def get_long_term():
            return await self.memory_handler.search_similar(
                query=query,
                limit=10,
                filter_dict={"memory_type": MemoryType.LONG_TERM}
            )

        async def get_reflection():
            return await self.memory_handler.search_similar(
                query=query,
                limit=5,
                filter_dict={"memory_type": MemoryType.REFLECTION}
            )

        # Gather all memory fetching tasks
        short_term, long_term, reflection = await asyncio.gather(
            get_short_term(),
            get_long_term(),
            get_reflection()
        )

        # Combine and score all memories
        all_memories = []
        for memory in short_term + long_term + reflection:
            decay_score = self._calculate_decay_score(memory)
            relevance_score = memory.get('score', 0.5)  # Default to 0.5 for short-term
            final_score = (decay_score + relevance_score) / 2
            memory['final_score'] = final_score
            all_memories.append(memory)

        # Sort by final score and filter low scores
        scored_memories = sorted(all_memories, key=lambda x: x['final_score'], reverse=True)
        filtered_memories = [m for m in scored_memories if m['final_score'] > 0.3]

        # Trim to fit token window
        current_tokens = 0
        final_memories = []
        for memory in filtered_memories:
            tokens = self._estimate_tokens(memory['content'])
            if current_tokens + tokens <= self.token_window:
                final_memories.append(memory)
                current_tokens += tokens
            else:
                break

        # Format and return the context
        return self._format_memories_by_type(final_memories)
