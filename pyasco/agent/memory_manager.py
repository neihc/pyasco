from typing import Dict, Any, Optional, List
import math
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
from dataclasses import dataclass
from ..services.llm import LLMService
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType

class MemoryManager:
    """Manages memory operations for the agent using LanceDB"""
    
    def __init__(self, memory_handler: LanceDBMemoryHandler, token_window: int = 2000):
        self.memory_handler = memory_handler
        self.token_window = token_window
        self.decay_threshold_days = 7  # Days after which to consider decay
        self.relevance_threshold = 0.3  # Minimum relevance score to keep in short-term

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
        decay_score = math.exp(-age / (7 * 24 * 3600))
        
        # Frequency score based on access count
        access_count = memory.get('access_count', 0)
        frequency_score = 1 - math.exp(-access_count / 5)  # Saturates around 15 accesses
        
        # Weights for different factors
        weights = {
            'decay': 0.3,      # Recent memories
            'relevance': 0.4,  # Search relevance
            'frequency': 0.3   # Access frequency
        }
        
        # Calculate weighted sum
        final_score = (
            weights['decay'] * decay_score +
            weights['relevance'] * relevance_score +
            weights['frequency'] * frequency_score
        )
        
        return final_score

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

    async def _decay_short_term_memories(self):
        """
        Move old or irrelevant short-term memories to long-term storage
        """
        # Get all short-term memories sorted by creation time
        short_term_memories = await self.memory_handler.search_similar(
            query="",
            limit=100,  # Reasonable batch size
            filter_dict=f"memory_type = '{MemoryType.SHORT_TERM}'",
            sort_by="created_at",
            ascending=True
        )

        if not short_term_memories:
            return

        # Get the most recent memory for relevance comparison
        latest_memory = await self.memory_handler.search_similar(
            query="",
            limit=1,
            filter_dict={"memory_type": MemoryType.SHORT_TERM},
            sort_by="created_at",
            ascending=False
        )
        latest_content = latest_memory[0]['content'] if latest_memory else ""

        decay_threshold = datetime.now() - timedelta(days=self.decay_threshold_days)
        
        for memory in short_term_memories:
            should_decay = False
            created_at = memory['created_at']
            
            # Check time-based decay
            if created_at < decay_threshold:
                should_decay = True
            
            # Check relevance-based decay
            if not should_decay and latest_content:
                relevance = await self.memory_handler.search_similar(
                    query=latest_content,
                    limit=1,
                    filter_dict=f"id = '{memory['id']}'"
                )
                if relevance and relevance[0]['score'] < self.relevance_threshold:
                    should_decay = True
            
            if should_decay:
                # Move to long-term memory
                memory_data = {
                    'content': memory['content'],
                    'memory_type': MemoryType.LONG_TERM,
                    'metadata': memory['metadata'],
                    'tags': memory['tags']
                }
                
                # Create new long-term memory
                await self.memory_handler.add_memory(memory_data)
                
                # Delete the short-term memory
                await self.memory_handler.delete_memory(memory['id'])

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
        
        memory_id = await self.memory_handler.add_memory(memory_data)
        
        # Trigger decay process after adding new memory
        await self._decay_short_term_memories()
        
        return memory_id

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
                filter_dict=f"memory_type = '{MemoryType.LONG_TERM}'"
            )

        async def get_reflection():
            return await self.memory_handler.search_similar(
                query=query,
                limit=5,
                filter_dict=f"memory_type = '{MemoryType.REFLECTION}'"
            )

        # Gather all memory fetching tasks
        short_term, long_term, reflection = await asyncio.gather(
            get_short_term(),
            get_long_term(),
            get_reflection()
        )

        # Score all memories using combined factors
        all_memories = []
        for memory in short_term + long_term + reflection:
            relevance_score = memory.get('score', 0.5)  # Default to 0.5 for short-term
            final_score = self._calculate_memory_score(memory, relevance_score)
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
