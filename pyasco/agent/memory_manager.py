from typing import Dict, Any, Optional, List
import math
from datetime import datetime, timezone
from collections import defaultdict
import asyncio
import logging
import os
from pathlib import Path
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
                 token_window: int = 10000):
        self.memory_handler = memory_handler
        self.llm_service = llm_service
        self.token_window = token_window
        
        # Ensure logs directory exists
        log_dir = Path.home() / '.pyasco' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger with path to ~/.pyasco/logs/
        self.logger = setup_logger('memory_manager', log_file='memory_manager.log')
        self.logger.info(f"Initializing MemoryManager with token window: {token_window}")
        
        self.decay_handler = MemoryDecayHandler(
            memory_handler=memory_handler,
            llm_service=llm_service,
        )
        self.logger.debug("Memory decay handler initialized")

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
        now = datetime.now().astimezone(timezone.utc)
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
        self.logger.debug(f"Storing new memory with content length: {len(content)}")
        self.logger.debug(f"Memory meta {meta}")
        
        memory_data = {
            'content': content,
            'memory_type': MemoryType.SHORT_TERM.value,
            'metadata': meta or {},
            'tags': []  # Could be enhanced to extract relevant tags
        }
        
        try:
            memory_id = await self.memory_handler.add_memory(memory_data)
            self.logger.info(f"Successfully created new memory with ID: {memory_id}")
            return memory_id
        except Exception as e:
            self.logger.error(f"Failed to create memory: {str(e)}", exc_info=True)
            raise

    async def get_context(self, query: str) -> str:
        """
        Get relevant context from different types of memories
        
        Args:
            query: The query to search memories with
            
        Returns:
            str: Formatted context from relevant memories
        """
        self.logger.info(f"Getting context for query: '{query[:50]}...' (truncated)")

        # Fetch different types of memories in parallel
        async def get_short_term():
            try:
                self.logger.debug("Fetching short-term memories")
                # First get latest short term memories using SQL
                sql_query = """
                SELECT *
                FROM memories 
                WHERE memory_type = 'short_term'
                ORDER BY created_at DESC
                LIMIT 20
                """
                self.logger.debug(f"Executing SQL query: {sql_query}")
                memories = await self.memory_handler.sql_query(sql_query)
                
                if not memories:
                    self.logger.info("No short-term memories found")
                    return []
                
                self.logger.debug(f"Found {len(memories)} short-term memories")
                    
                # Then query again with these IDs to get relevance scores
                memory_ids = [f"'{m['id']}'" for m in memories]
                id_filter = f"id IN ({', '.join(memory_ids)})"
                
                self.logger.debug(f"Searching for similar memories with filter: {id_filter}")
                scored_memories = await self.memory_handler.search_similar(
                    query=query,  # Now use the actual query
                    limit=len(memory_ids),
                    filter_dict=id_filter,
                )
                
                self.logger.debug(f"Retrieved {len(scored_memories)} scored short-term memories")
                return scored_memories
                
            except Exception as e:
                self.logger.error(f"Error fetching short-term memories: {e}", exc_info=True)
                return []

        async def get_long_term():
            try:
                self.logger.debug("Fetching long-term memories")
                long_term_memories = await self.memory_handler.search_similar(
                    query=query,
                    limit=10,
                    filter_dict=f"memory_type = '{MemoryType.LONG_TERM.value}'"
                )
                self.logger.debug(f"Retrieved {len(long_term_memories)} long-term memories")
                return long_term_memories
            except Exception as e:
                self.logger.error(f"Error fetching long-term memories: {e}", exc_info=True)
                return []

        async def get_reflection():
            try:
                self.logger.debug("Fetching reflection memories")
                reflection_memories = await self.memory_handler.search_similar(
                    query=query,
                    limit=5,
                    filter_dict=f"memory_type = '{MemoryType.REFLECTION.value}'"
                )
                self.logger.debug(f"Retrieved {len(reflection_memories)} reflection memories")
                return reflection_memories
            except Exception as e:
                self.logger.error(f"Error fetching reflection memories: {e}", exc_info=True)
                return []

        try:
            self.logger.info("Gathering memories from all sources")
            # Gather all memory fetching tasks
            short_term, long_term = await asyncio.gather(
                get_short_term(),
                get_long_term(),
            )
            self.logger.info(f"Successfully gathered memories: {len(short_term)} short-term, {len(long_term)} long-term")
        except Exception as e:
            self.logger.error(f"Error gathering memories: {e}", exc_info=True)
            short_term, long_term = [], []

        # Score all memories using combined factors
        self.logger.debug("Calculating memory scores")
        all_memories = []
        for memory in short_term + long_term:
            relevance_score = memory.get('_relevance_score', 0.5)  # Default to 0.5 for short-term
            final_score = self._calculate_memory_score(memory, relevance_score)
            memory['final_score'] = final_score
            all_memories.append(memory)
            self.logger.debug(f"Memory {memory['id'][:8]}... scored {final_score:.4f}")

        # First get X most recent memories
        recent_count = 10  # X recent memories
        scored_count = 5  # Y scored memories
        
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

        self.logger.info("Selected memories with scores:")
        for memory in final_memories:
            self.logger.info(f"Score: {memory['final_score']:.3f} | Type: {memory['memory_type']} | ID: {memory['id'][:8]}... | Content: {memory['content'][:100]}...")

        self.logger.info(f"Retrieved {len(final_memories)} relevant memories")
        self.logger.debug(f"Memory scores: {[m['final_score'] for m in final_memories]}")
        self.logger.debug(f"Memory types: {[m['memory_type'] for m in final_memories]}")
        
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
            # Get count of short-term memories before decay
            count_query = "SELECT COUNT(*) as count FROM memories WHERE memory_type = 'short_term'"
            result = await self.memory_handler.sql_query(count_query)
            before_count = result[0]['count'] if result else 0
            self.logger.info(f"Short-term memory count before decay: {before_count}")
            
            # Trigger the decay process
            start_time = datetime.now()
            self.logger.info(f"Decay process started at: {start_time.isoformat()}")
            
            await self.decay_handler.decay_short_term_memories()
            
            # Get count after decay
            result = await self.memory_handler.sql_query(count_query)
            after_count = result[0]['count'] if result else 0
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Memory decay process completed in {duration:.2f} seconds")
            self.logger.info(f"Short-term memory count after decay: {after_count}")
            self.logger.info(f"Memories processed: {before_count - after_count}")
        except Exception as e:
            self.logger.error(f"Error during memory decay process: {e}", exc_info=True)
            self.logger.error(f"Stack trace: ", exc_info=True)
