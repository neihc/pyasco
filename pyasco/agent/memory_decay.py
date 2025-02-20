from datetime import datetime, timedelta
from typing import Dict, Any, List
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType

class MemoryDecayHandler:
    """Handles the decay of short-term memories to long-term storage"""
    
    def __init__(self, memory_handler: LanceDBMemoryHandler, 
                 decay_threshold_days: int = 7,
                 relevance_threshold: float = 0.3):
        self.memory_handler = memory_handler
        self.decay_threshold_days = decay_threshold_days
        self.relevance_threshold = relevance_threshold

    async def decay_short_term_memories(self):
        """
        Move old or irrelevant short-term memories to long-term storage
        """
        # Get all short-term memories sorted by creation time
        short_term_memories = await self.memory_handler.search_similar(
            query="",
            limit=100,  # Reasonable batch size
            filter_dict=f"memory_type = '{MemoryType.SHORT_TERM.value}'",
            sort_by="created_at",
            ascending=True
        )

        if not short_term_memories:
            return

        # Get the most recent memory for relevance comparison
        latest_memory = await self.memory_handler.search_similar(
            query="",
            limit=1,
            filter_dict={"memory_type": MemoryType.SHORT_TERM.value},
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
                if relevance and relevance[0]['_relevance_score'] < self.relevance_threshold:
                    should_decay = True
            
            if should_decay:
                # Move to long-term memory
                memory_data = {
                    'content': memory['content'],
                    'memory_type': MemoryType.LONG_TERM.value,
                    'metadata': memory['metadata'],
                    'tags': memory['tags']
                }
                
                # Create new long-term memory
                await self.memory_handler.add_memory(memory_data)
                
                # Delete the short-term memory
                await self.memory_handler.delete_memory(memory['id'])
