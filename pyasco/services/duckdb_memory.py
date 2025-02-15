from typing import List, Dict, Any, Optional
import json
import duckdb
import numpy as np
from datetime import datetime
from pathlib import Path

from ..services.llm import LLMService
from ..services.embedding import EmbeddingService

class DuckDBMemoryHandler:
    """Handler for processing and storing memories using DuckDB"""
    
    def __init__(self, 
                 llm_service: LLMService,
                 embedding_service: EmbeddingService,
                 db_path: str = "~/.pyasco/memories.db"):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database schema with VSS extension support"""
        # Install and load VSS extension
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")
        self.conn.execute("SET hnsw_enable_experimental_persistence=true;")
        
        # Create table with FLOAT[] type for embeddings
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                embedding FLOAT[1024] NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create HNSW index for fast similarity search
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS memory_embedding_idx 
            ON memories 
            USING HNSW (embedding);
        """)

    def remember(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract meaningful memories from content and store them in the database
        
        Args:
            content: The text content to extract memories from
            context: Optional contextual metadata
            
        Returns:
            Dict containing status and extracted memories
        """
        # Use LLM to extract meaningful memories
        prompt = f"""Extract independent meaningful memories from the following content. 
        Each memory should be self-contained and meaningful on its own.
        Return the memories as a JSON array of strings.
        
        Content: {content}"""
        
        response = self.llm_service.get_response([
            {"role": "system", "content": "You are a helpful assistant that extracts meaningful memories from text."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            memories = json.loads(response.content)
            if not isinstance(memories, list):
                raise ValueError("Expected JSON array of memories")
        except (json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Failed to parse memories from LLM response: {e}")

        stored_memories = []
        for memory in memories:
            # Generate embedding
            embedding = self.embedding_service.get_embedding(memory)
            
            # Store in database with embedding as FLOAT array
            self.conn.execute("""
                INSERT INTO memories (content, embedding, metadata)
                VALUES (?, ?::FLOAT[], ?);
            """, [memory, embedding.tolist(), json.dumps(context or {})])
            
            stored_memories.append({
                "content": memory,
                "metadata": context
            })
            
        return {
            "status": "success",
            "memories": stored_memories
        }

    def recall(self, query: str, limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Recall relevant memories based on semantic similarity
        
        Args:
            query: The query text to find relevant memories
            limit: Maximum number of memories to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant memories with their metadata
        """
        query_embedding = self.embedding_service.get_embedding(query)
        
        results = self.conn.execute("""
            SELECT 
                content,
                meta:JSON as metadata,
                1 - array_distance(embedding, ?::FLOAT[]) as similarity,
                created_at
            FROM memories
            WHERE 1 - array_distance(embedding, ?::FLOAT[]) >= ?
            ORDER BY array_distance(embedding, ?::FLOAT[])
            LIMIT ?;
        """, [
            query_embedding.tolist(),
            query_embedding.tolist(),
            similarity_threshold,
            query_embedding.tolist(),
            limit
        ]).fetchall()
        
        return [{
            "content": row[0],
            "metadata": json.loads(row[1]),
            "similarity": float(row[2]),
            "created_at": row[3].isoformat()
        } for row in results]

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
