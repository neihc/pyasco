from typing import Dict, Any, Optional, List
import math
from datetime import datetime
from collections import defaultdict
import asyncio
from dataclasses import dataclass
from ..services.llm import LLMService
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType
from .memory_decay import MemoryDecayHandler

class MemoryManager:
    """Manages memory operations for the agent using LanceDB"""
    
    def __init__(self, 
                 memory_handler: LanceDBMemoryHandler, 
                 llm_service: LLMService,
                 token_window: int = 2000):
        self.memory_handler = memory_handler
        self.llm_service = llm_service
        self.token_window = token_window
        self.decay_handler = MemoryDecayHandler(
            memory_handler=memory_handler,
            llm_service=llm_service,
        )

    def _calculate_memory_score(self, memory: Dict[str, Any], relevance_score: float = 0.5) -> float:
        """
        Calculate combined memory score based on multiple factors
        
        Args:
            memory: Memory dictionary containing metadata
            relevance_score: Similarity/relevance score from search (default 0.5)
            
        Returns:
            float: Combined score between 0 and 1
        """
        # Time decay score (7 days half-life)
        now = datetime.now()
        age = (now - memory['created_at']).total_seconds()
        decay_score = math.exp(-age / (3600))
        
        # Frequency score based on access count
        access_count = memory.get('access_count', 0)
        frequency_score = 1 - math.exp(-access_count / 5)  # Saturates around 15 accesses
        
        # Get importance score from memory or default to 0.5
        importance_score = memory.get('importance_score', 0.5)
        
        # Weights for different factors
        weights = {
            'decay': 0.35,      # Recent memories
            'relevance': 0.3,   # Search relevance
            'frequency': 0.2,   # Access frequency
            'importance': 0.15  # Explicit importance
        }
        
        # Calculate weighted sum
        final_score = (
            weights['decay'] * decay_score +
            weights['relevance'] * relevance_score +
            weights['frequency'] * frequency_score +
            weights['importance'] * importance_score
        )
        
        return final_score

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimate of token count"""
        return len(text.split()) * 1.3  # Rough approximation

    async def _format_memories_by_type(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories grouped by type"""
        # Increment access count for all memories being accessed
        memory_ids = [memory['id'] for memory in memories]
        await self.memory_handler.increment_access_count(memory_ids)
        
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
            'memory_type': MemoryType.SHORT_TERM.value,
            'metadata': meta or {},
            'tags': []  # Could be enhanced to extract relevant tags
        }
        
        memory_id = await self.memory_handler.add_memory(memory_data)
        return memory_id

    async def get_context(self, query: str) -> str:
        """
        Get relevant context from different types of memories
        
        Args:
            query: The query to search memories with
            
        Returns:
            str: Formatted context from relevant memories
        """
        import logging
        logger = logging.getLogger(__name__)

        # Fetch different types of memories in parallel
        async def get_short_term():
            try:
                # First get latest short term memories
                memories = await self.memory_handler.search_similar(
                    query="",  # Empty query to get latest
                    limit=10,
                    filter_dict={"memory_type": MemoryType.SHORT_TERM.value},
                    sort_by='created_at',
                    ascending=False
                )
                
                if not memories:
                    return []
                    
                # Then query again with these IDs to get relevance scores
                memory_ids = [f"'{m['id']}'" for m in memories]
                id_filter = f"id IN ({', '.join(memory_ids)})"
                
                scored_memories = await self.memory_handler.search_similar(
                    query=query,  # Now use the actual query
                    limit=len(memory_ids),
                    filter_dict=id_filter
                )
                
                return scored_memories
                
            except Exception as e:
                logger.error(f"Error fetching short-term memories: {e}")
                return []

        async def get_long_term():
            try:
                return await self.memory_handler.search_similar(
                    query=query,
                    limit=10,
                    filter_dict=f"memory_type = '{MemoryType.LONG_TERM.value}'"
                )
            except Exception as e:
                logger.error(f"Error fetching long-term memories: {e}")
                return []

        async def get_reflection():
            try:
                return await self.memory_handler.search_similar(
                    query=query,
                    limit=5,
                    filter_dict=f"memory_type = '{MemoryType.REFLECTION.value}'"
                )
            except Exception as e:
                logger.error(f"Error fetching reflection memories: {e}")
                return []

        try:
            # Gather all memory fetching tasks
            short_term, long_term, reflection = await asyncio.gather(
                get_short_term(),
                get_long_term(),
                get_reflection()
            )
            import pdb; pdb.set_trace()
        except Exception as e:
            logger.error(f"Error gathering memories: {e}")
            short_term, long_term, reflection = [], [], []

        # Score all memories using combined factors
        all_memories = []
        for memory in short_term + long_term + reflection:
            relevance_score = memory.get('_relevance_score', 0.5)  # Default to 0.5 for short-term
            final_score = self._calculate_memory_score(memory, relevance_score)
            memory['final_score'] = final_score
            all_memories.append(memory)

        # Sort by final score and filter low scores
        scored_memories = sorted(all_memories, key=lambda x: x['final_score'], reverse=True)
        filtered_memories = [m for m in scored_memories if m['final_score'] > 0.2]

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

        logger.info("Selected memories with scores:")
        for memory in final_memories:
            logger.info(f"Score: {memory['final_score']:.3f} | Content: {memory['content'][:100]}...")

        # Format and return the context
        return await self._format_memories_by_type(final_memories)

    async def trigger_decay(self) -> None:
        """
        Trigger the memory decay process to consolidate short-term memories into long-term memories.
        This process:
        1. Identifies old or frequently accessed short-term memories
        2. Clusters related memories
        3. Uses LLM to process and integrate them into long-term memories
        4. Removes processed short-term memories
        """
        try:
            await self.decay_handler.decay_short_term_memories()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error during memory decay process: {e}")
