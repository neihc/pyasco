from typing import List, Dict, Any, Optional
from enum import Enum
import json
import lancedb
import numpy as np
from datetime import datetime
from pathlib import Path
import uuid
import pandas as pd
import pyarrow as pa
from lancedb.pydantic import Vector, LanceModel
from lancedb.embeddings import get_registry
from lancedb.rerankers import JinaReranker
import os

from ..services.llm import LLMService
from ..services.code_snippet_extractor import CodeSnippetExtractor

# Get Jina embedding function
jina_embed = get_registry().get("jina").create(
    name="jina-embeddings-v3"
)

class MemoryType(str, Enum):
    """Types of memories that can be stored"""
    SHORT_TERM = "short_term"    # Recent, temporary memories
    LONG_TERM = "long_term"      # Important, permanent memories
    REFLECTION = "reflection"     # Meta-cognitive memories about learning and understanding
    PROCEDURAL = "procedural"    # Task-related memories about how to do things

class Memory(LanceModel):
    """Pydantic model for memory table schema"""
    id: str
    content: str = jina_embed.SourceField()
    vector: Vector(1024) = jina_embed.VectorField()
    memory_type: MemoryType
    metadata: str  # JSON string
    tags: List[str]
    created_at: datetime
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    event_time: Optional[datetime] = None
    access_count: int = 0

class LanceDBMemoryHandler:
    """Handler for processing and storing memories using LanceDB"""
    
    def __init__(self, 
                 db_path: str = "~/.pyasco/memories",
                 jina_api_key: Optional[str] = None):
        self.code_extractor = CodeSnippetExtractor()
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup Jina API key if provided
        if jina_api_key:
            os.environ['JINA_API_KEY'] = jina_api_key
            
        # Initialize reranker
        self.reranker = JinaReranker(api_key=jina_api_key) if jina_api_key else None
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self.Memory = Memory  # Use the globally defined Memory class
        self._initialize_db()


    async def _initialize_db(self):
        """Initialize the database table with vector search and full-text search support"""
        if "memories" not in self.db.table_names():
            table = self.db.create_table("memories", schema=self.Memory, mode="create")
            # Create full-text search index on content for hybrid search
            await table.create_fts_index(["content"], replace=True)

    async def add_memory(self, 
                        memory_data: Dict[str, Any],
                        memory_id: Optional[str] = None) -> str:
        """
        Add a new memory to the database.
        
        Args:
            memory_data: Dictionary containing memory data
            memory_id: Optional memory ID (UUID generated if not provided)
            
        Returns:
            str: ID of the created memory
            
        The memory_data dict should contain:
            - content: str (required)
            - memory_type: MemoryType (required)
            - metadata: dict (will be converted to JSON)
            - tags: List[str]
            - valid_from: datetime (optional)
            - valid_until: datetime (optional) 
            - event_time: datetime (optional)
        """
        # Generate UUID if not provided
        memory_id = memory_id or str(uuid.uuid4())
        
        # Ensure required fields
        if 'content' not in memory_data:
            raise ValueError("Memory content is required")
        if 'memory_type' not in memory_data:
            raise ValueError("Memory type is required")
            
        # Convert metadata dict to JSON string if needed
        if 'metadata' in memory_data and isinstance(memory_data['metadata'], dict):
            memory_data['metadata'] = json.dumps(memory_data['metadata'])
        
        # Create memory record
        memory = Memory(
            id=memory_id,
            content=memory_data['content'],
            memory_type=memory_data['memory_type'],
            metadata=memory_data.get('metadata', '{}'),
            tags=memory_data.get('tags', []),
            created_at=datetime.now(),
            valid_from=memory_data.get('valid_from'),
            valid_until=memory_data.get('valid_until'),
            event_time=memory_data.get('event_time')
        )
        
        # Add to database
        table = self.db.open_table("memories")
        await table.add([memory])
        
        return memory_id

    async def search_similar(self, 
                           query: str, 
                           limit: int = 5,
                           score_threshold: float = 0.0,
                           filter_dict: Optional[Dict] = None,
                           sort_by: Optional[str] = None,
                           ascending: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar memories using hybrid search (vector + text) with reranking.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_dict: Optional dictionary of filters to apply (e.g., {"memory_type": "long_term"})
            sort_by: Optional field name to sort results by
            ascending: Sort direction (True for ascending, False for descending)
            
        Returns:
            List of memory dictionaries with similarity scores
        """
        table = self.db.open_table("memories")
        
        # Perform hybrid search with optional filter
        search = table.search(query, query_type="hybrid")
        if filter_dict:
            search = search.where(filter_dict)
            
        # Apply sorting if specified
        if sort_by:
            search = search.sort(sort_by, ascending=ascending)
        
        # Apply reranker if available
        if self.reranker:
            results = await search.rerank(reranker=self.reranker).limit(limit).to_list()
        else:
            results = await search.limit(limit).to_list()
            
        # Filter by score threshold and convert to dicts
        filtered_results = []
        for result in results:
            if result.score >= score_threshold:
                memory_dict = {
                    "id": result.id,
                    "content": result.content,
                    "memory_type": result.memory_type,
                    "metadata": json.loads(result.metadata),
                    "tags": result.tags,
                    "created_at": result.created_at,
                    "score": result.score
                }
                if result.valid_from:
                    memory_dict["valid_from"] = result.valid_from
                if result.valid_until:
                    memory_dict["valid_until"] = result.valid_until
                if result.event_time:
                    memory_dict["event_time"] = result.event_time
                    
                filtered_results.append(memory_dict)
                
        return filtered_results

    async def increment_access_count(self, memory_ids: List[str]) -> None:
        """
        Increment the access count for multiple memories.
        
        Args:
            memory_ids: List of memory IDs to update
        """
        table = self.db.open_table("memories")
        
        # Build ID filter
        id_filter = " OR ".join([f"id = '{mid}'" for mid in memory_ids])
        
        # Get current memories
        current = await table.where(id_filter).to_list()
        if not current:
            return  # Skip if no memories found
            
        # Update access counts
        updated_memories = []
        for memory in current:
            updated_memory = Memory(
                **{**memory.dict(),
                   "access_count": memory.access_count + 1}
            )
            updated_memories.append(updated_memory)
        
        # Replace the records
        await table.delete(id_filter)
        await table.add(updated_memories)
