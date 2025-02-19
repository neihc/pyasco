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
