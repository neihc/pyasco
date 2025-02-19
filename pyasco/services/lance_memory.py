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
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self.Memory = Memory  # Use the globally defined Memory class
        self._initialize_db()


    def _initialize_db(self):
        """Initialize the database table with vector search and full-text search support"""
        if "memories" not in self.db.table_names():
            table = self.db.create_table("memories", schema=self.Memory, mode="create")
            # Create full-text search index on content
            table.create_fts_index(["content"])

    def add_memory(self, 
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
        table.add([memory])
        
        return memory_id
