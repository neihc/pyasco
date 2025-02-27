from typing import Dict, Any, Optional, List
import math
from datetime import datetime
from collections import defaultdict
import asyncio
import logging
from dataclasses import dataclass
from ..services.llm import LLMService
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType
from .memory_decay import MemoryDecayHandler
from ..logger_config import setup_logger

class MemoryManager:
    """Manages memory operations for the agent using LanceDB"""
    
    def __init__(self, 
                 memory_handler: LanceDBMemoryHandler, 
                 llm_service: LLMService,
                 token_window: int = 2000):
        self.memory_handler = memory_handler
        self.llm_service = llm_service
        self.token_window = token_window
        self.logger = setup_logger('memory_manager', log_file='memory.log')
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
            'relevance': 0.15,   # Search relevance
            'frequency': 0.15,   # Access frequency
            'importance': 0.25  # Explicit importance
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
        
        # Group memories by type
        grouped = defaultdict(list)
        for memory in memories:
            grouped[memory['memory_type']].append(memory)

        sections = []
        
        # Format short-term memories (conversation) - sorted by created_at ascending
        if MemoryType.SHORT_TERM in grouped:
            sections.append("<current conversation>")
            # Sort by created_at in ascending order
            sorted_memories = sorted(grouped[MemoryType.SHORT_TERM], 
                                    key=lambda x: x['created_at'])
            for memory in sorted_memories:
                sections.append(memory['content'])
            sections.append("</current conversation>")
            sections.append("")  # Empty line between sections
        
        # Format long-term memories (reference) with dividers
        if MemoryType.LONG_TERM in grouped:
            sections.append("<reference memory>")
            for i, memory in enumerate(grouped[MemoryType.LONG_TERM]):
                if i > 0:
                    sections.append("------")  # Divider between references
                sections.append(memory['content'])
            sections.append("</reference memory>")
            sections.append("")  # Empty line between sections
            
        # Format reflection memories if any
        if MemoryType.REFLECTION in grouped:
            sections.append("<reflection>")
            for memory in grouped[MemoryType.REFLECTION]:
                sections.append(memory['content'])
            sections.append("</reflection>")
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
        self.logger.info(f"Created new memory with ID: {memory_id}")
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
                # First get latest short term memories using SQL
                query = """
                SELECT *
                FROM memories 
                WHERE memory_type = 'short_term'
                ORDER BY created_at DESC
                LIMIT 20
                """
                memories = await self.memory_handler.sql_query(query)
                import pdb; pdb.set_trace()
                
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
            short_term, long_term = await asyncio.gather(
                get_short_term(),
                get_long_term(),
            )
        except Exception as e:
            logger.error(f"Error gathering memories: {e}")
            short_term, long_term = [], []

        # Score all memories using combined factors
        all_memories = []
        for memory in short_term + long_term:
            relevance_score = memory.get('_relevance_score', 0.5)  # Default to 0.5 for short-term
            final_score = self._calculate_memory_score(memory, relevance_score)
            memory['final_score'] = final_score
            all_memories.append(memory)

        # First get X most recent memories
        recent_count = 5  # X recent memories
        scored_count = 10  # Y scored memories
        
        # Sort by timestamp for recent memories
        recent_memories = sorted(short_term, key=lambda x: x['created_at'], reverse=True)[:recent_count]
        recent_ids = {m['id'] for m in recent_memories}
        
        # Sort remaining memories by score and filter low scores
        remaining_memories = [m for m in all_memories if m['id'] not in recent_ids]
        scored_memories = sorted(remaining_memories, key=lambda x: x['final_score'], reverse=True)
        filtered_memories = [m for m in scored_memories if m['final_score'] > 0.3][:scored_count]

        # Combine recent and scored memories
        final_memories = recent_memories + filtered_memories

        # Trim to fit token window if needed
        current_tokens = 0
        token_limited_memories = []
        for memory in final_memories:
            tokens = self._estimate_tokens(memory['content'])
            if current_tokens + tokens <= self.token_window:
                token_limited_memories.append(memory)
                current_tokens += tokens
            else:
                break
                
        final_memories = token_limited_memories

        logger.info("Selected memories with scores:")
        for memory in final_memories:
            logger.info(f"Score: {memory['final_score']:.3f} | Content: {memory['content'][:100]}...")

        self.logger.info(f"Retrieved {len(final_memories)} relevant memories")
        self.logger.debug(f"Memory scores: {[m['final_score'] for m in final_memories]}")
        
        # Format and return the context
        formatted_context = await self._format_memories_by_type(final_memories)
        self.logger.debug(f"Formatted context length: {len(formatted_context)}")
        return formatted_context

    async def trigger_decay(self) -> None:
        """
        Trigger the memory decay process to consolidate short-term memories into long-term memories.
        This process:
        1. Identifies old or frequently accessed short-term memories
        2. Clusters related memories
        3. Uses LLM to process and integrate them into long-term memories
        4. Removes processed short-term memories
        """
        self.logger.info("Starting memory decay process")
        try:
            await self.decay_handler.decay_short_term_memories()
            self.logger.info("Memory decay process completed successfully")
        except Exception as e:
            self.logger.error(f"Error during memory decay process: {e}", exc_info=True)
