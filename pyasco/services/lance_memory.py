import os
from typing import List, Dict, Any, Optional
from enum import Enum
import json
from datetime import datetime
from pathlib import Path
import uuid
import duckdb
import lancedb
from lancedb.pydantic import Vector, LanceModel
from lancedb.embeddings import get_registry
from lancedb.rerankers import JinaReranker

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
    summary: Optional[str] = None
    vector: Vector(1024) = jina_embed.VectorField()
    memory_type: str
    metadata: str  # JSON string
    tags: List[str]
    created_at: datetime
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    event_time: Optional[datetime] = None
    access_count: int = 0
    importance_score: float = 0.0

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
        self.reranker = JinaReranker(
            api_key=os.environ['JINA_API_KEY'],
            column="content"
        ) if os.environ['JINA_API_KEY'] else None
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self.Memory = Memory  # Use the globally defined Memory class
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database table with vector search and full-text search support"""
        if "memories" not in self.db.table_names():
            table = self.db.create_table("memories", schema=self.Memory, mode="create")
            # Create full-text search index on content for hybrid search
            table.create_fts_index(["content"], replace=True)

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
        
        # Create memory record with vector embedding
        memory_dict = {
            'id': memory_id,
            'content': memory_data['content'],
            'memory_type': memory_data['memory_type'],
            'metadata': memory_data.get('metadata', '{}'),
            'tags': memory_data.get('tags', []),
            'created_at': datetime.now(),
            'valid_from': memory_data.get('valid_from'),
            'valid_until': memory_data.get('valid_until'),
            'event_time': memory_data.get('event_time'),
            'access_count': 0,
            'summary': '',
            'importance_score': memory_data.get('importance_score', 0),
        }
        
        # Add to database
        table = self.db.open_table("memories")
        table.add([memory_dict])
        
        return memory_id

    async def search_similar(self, 
                           query: str, 
                           limit: int = 5,
                           score_threshold: float = 0.0,
                           filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar memories using hybrid search (vector + text) with reranking.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_dict: Optional dictionary of filters to apply (e.g., {"memory_type": "long_term"})
            
        Returns:
            List of memory dictionaries with similarity scores
        """
        table = self.db.open_table("memories")
        
        try:
            # Perform hybrid search with optional filter
            search = table.search(query, query_type="hybrid").limit(limit)
            if filter_dict:
                search = search.where(filter_dict if isinstance(filter_dict, str) else " AND ".join(f"{k} = '{v}'" for k, v in filter_dict.items()))
                
            if self.reranker:
                search = search.rerank(reranker=self.reranker)

            # Get all results as pandas DataFrame
            df = search.to_pandas()
        except ValueError as e:
            if "not enough values to unpack" in str(e):
                # Return empty DataFrame if no results found
                import pandas as pd
                df = pd.DataFrame(columns=['id', 'content', 'memory_type', 'metadata', 'tags', 
                                         'created_at', 'valid_from', 'valid_until', 'event_time',
                                         'access_count', 'importance_score', 'summary', '_relevance_score'])
            else:
                raise
        
        # Apply limit and filter by score threshold
        df = df.head(limit)
        df = df[df._relevance_score >= score_threshold]
        
        # Convert metadata from JSON strings to dicts
        df['metadata'] = df['metadata'].apply(json.loads)
        
        # Convert to dict records, keeping all fields
        results = df.to_dict('records')
        
        return results

    async def increment_access_count(self, memory_ids: List[str]) -> None:
        """
        Increment the access count for multiple memories.
        
        Args:
            memory_ids: List of memory IDs to update
        """
        if not memory_ids:
            return
        table = self.db.open_table("memories")
        
        # Build ID filter with proper parentheses
        id_conditions = [f"id = '{mid}'" for mid in memory_ids]
        id_filter = f"({' OR '.join(id_conditions)})"
        
        # Get current memories
        current = table.search().where(id_filter).to_list()
        if not current:
            return  # Skip if no memories found
            
        # Update access counts using merge_insert
        updated_memories = []
        for memory in current:
            # Create a clean copy without vector field
            memory_copy = {k: v for k, v in memory.items() if k != 'vector'}
            memory_copy['access_count'] = memory_copy.get('access_count', 0) + 1
            updated_memories.append(memory_copy)
        
        # Use merge_insert for atomic update
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(updated_memories)
        )

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by its ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if memory was found and deleted, False otherwise
        """
        table = self.db.open_table("memories")
        
        # Check if memory exists
        existing = table.search().where(f"id = '{memory_id}'").to_list()
        if not existing:
            return False
            
        # Delete the memory
        table.delete(f"id = '{memory_id}'")
        return True

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific fields of a memory.
        
        Args:
            memory_id: ID of the memory to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if memory was found and updated, False otherwise
        """
        table = self.db.open_table("memories")
        
        # Get existing memory
        existing = table.search().where(f"id = '{memory_id}'").to_list()
        if not existing:
            return False
            
        # Update memory with new values
        memory_dict = existing[0]
        memory_dict.update(updates)
        memory_dict['metadata'] = json.dumps(memory_dict.get('metadata', {}))
        memory_dict.pop('vector')
        
        # Use merge_insert for atomic update
        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([memory_dict])
        )
        
        return True

    async def sql_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a SQL query against the memories table using DuckDB.
        
        Args:
            query: SQL query string to execute
            
        Returns:
            List[Dict[str, Any]]: Query results as list of dictionaries
        """
        table = self.db.open_table("memories")
        arrow_table = table.to_lance()
        
        # Execute query and convert results to list of dicts
        result = duckdb.query(query, arrow_table)
        return result.fetchall()

    async def get_all_tags(self) -> List[str]:
        """
        Get all unique tags used across memories using pandas for efficiency.
        
        Returns:
            List[str]: List of unique tags
        """
        table = self.db.open_table("memories")
        
        # Get all memories as pandas DataFrame
        df = table.to_pandas()
        
        # Explode the tags column and get unique values
        if not df.empty and 'tags' in df.columns:
            all_tags = df['tags'].explode().dropna().unique().tolist()
            return sorted(all_tags)
        
        return []
