from typing import List, Dict, Any, Optional
import json
import duckdb
import numpy as np
from datetime import datetime
from pathlib import Path
import uuid

from ..services.llm import LLMService
from ..services.embedding import EmbeddingService
from ..services.code_snippet_extractor import CodeSnippetExtractor

class DuckDBMemoryHandler:
    """Handler for processing and storing memories using DuckDB"""
    
    def __init__(self, 
                 llm_service: LLMService,
                 embedding_service: EmbeddingService,
                 db_path: str = "~/.pyasco/memories.db"):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.code_extractor = CodeSnippetExtractor()
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
                id UUID PRIMARY KEY,
                content TEXT NOT NULL,
                embedding FLOAT[1024] NOT NULL,
                metadata JSON,
                tags TEXT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                valid_from TIMESTAMP,
                valid_until TIMESTAMP,
                event_time TIMESTAMP
            );
        """)
        
        # Create HNSW index for fast similarity search
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS memory_embedding_idx 
            ON memories 
            USING HNSW (embedding)
            WITH (metric = 'cosine');
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
        prompt = f"""
        Extract specific, concrete memories from the following content.
        Focus on distinct, actionable information and avoid generic or abstract concepts.
        Each memory should capture a single, well-defined piece of information.

        Guidelines:
        - Include specific details, numbers, dates, names, or actions
        - Break down complex information into individual memories
        - Exclude vague or general statements
        - Focus on factual, verifiable information
        - Identify temporal aspects (when the information is valid or occurred)
        - Assign relevant categorical tags

        CONTENT:
        {content}
        
        CONTEXT:
        {context or {}}
        
        Return a JSON object wrapped in a code block with this structure:
        ```json
        {{
            "memories": [
                {{
                    "content": "the specific memory with concrete details",
                    "type": "observation|fact|relationship",
                    "confidence": 0.0-1.0,
                    "tags": ["tag1", "tag2"],
                    "temporal": {{
                        "valid_from": "ISO timestamp or null",
                        "valid_until": "ISO timestamp or null",
                        "event_time": "ISO timestamp or null"
                    }},
                    "metadata": {{
                        "source": "original text",
                        "importance_level": 0.0-1.0,
                        "last_accessed": null,
                        "access_count": 0
                    }}
                }}
            ]
        }}
        ```
        """
        
        response = self.llm_service.get_response([{
            "role": "user",
            "content": prompt
        }])
        
        # Extract JSON using CodeSnippetExtractor
        snippets = self.code_extractor.extract_snippets(response)
        json_snippets = [s for s in snippets if s.language == "json"]
        
        if not json_snippets:
            raise ValueError("No JSON structure found in LLM response")
            
        try:
            memories_data = json.loads(json_snippets[0].content)
            if not isinstance(memories_data, dict) or "memories" not in memories_data:
                raise ValueError("Invalid memories structure in response")
            memories = memories_data["memories"]
        except (json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Failed to parse memories from LLM response: {e}")

        stored_memories = []
        for memory in memories:
            # Generate embedding for the memory content
            embedding = self.embedding_service.get_embedding(memory["content"])
            
            # Initialize metadata with defaults
            metadata = {
                "source": memory.get("metadata", {}).get("source", "unknown"),
                "importance_level": memory.get("metadata", {}).get("importance_level", 0.5),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0,
                "memory_type": memory.get("type", "observation"),
                "confidence": memory.get("confidence", 1.0)
            }
            
            # Add any additional context
            if context:
                metadata.update(context)
            
            # Store in database with embedding as FLOAT array
            memory_id = str(uuid.uuid4())
            # Extract temporal data
            temporal = memory.get("temporal", {})
            valid_from = temporal.get("valid_from")
            valid_until = temporal.get("valid_until")
            event_time = temporal.get("event_time")
            
            # Extract tags
            tags = memory.get("tags", [])
            
            self.conn.execute("""
                INSERT INTO memories (
                    id, content, embedding, metadata, tags,
                    valid_from, valid_until, event_time
                )
                VALUES (?, ?, ?::FLOAT[], ?, ?::TEXT[],
                        ?::TIMESTAMP, ?::TIMESTAMP, ?::TIMESTAMP);
            """, [
                memory_id, memory["content"], embedding[0].tolist(),
                json.dumps(metadata), tags,
                valid_from, valid_until, event_time
            ])
            
            stored_memories.append({
                "content": memory["content"],
                "metadata": metadata
            })
            
        return {
            "status": "success",
            "memories": stored_memories
        }

    def recall(self, query: str, limit: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Recall relevant memories based on semantic similarity
        
        Args:
            query: The query text to find relevant memories
            limit: Maximum number of memories to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant memories with their metadata
        """
        # Get embedding and ensure it's the right shape
        query_embedding = self.embedding_service.get_embedding(query)
        query_embedding = query_embedding[0].tolist()  # Flatten to 1D list
        
        results = self.conn.execute("""
            SELECT 
                content,
                metadata,
                array_cosine_distance(embedding, ?::FLOAT[1024]) as similarity,
                created_at,
                tags,
                valid_from,
                valid_until,
                event_time
            FROM memories
            WHERE array_cosine_distance(embedding, ?::FLOAT[1024]) >= ?
            ORDER BY array_cosine_distance(embedding, ?::FLOAT[1024]) DESC
            LIMIT ?;
        """, [
            query_embedding,
            query_embedding,
            similarity_threshold,
            query_embedding,
            limit
        ]).fetchall()
        
        memories = []
        for row in results:
            # Update access count and last_accessed time
            metadata = json.loads(row[1])
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            metadata["last_accessed"] = datetime.now().isoformat()
            
            # Update the metadata in the database
            self.conn.execute("""
                UPDATE memories 
                SET metadata = ? 
                WHERE content = ?
            """, [json.dumps(metadata), row[0]])
            
            memories.append({
                "content": row[0],
                "metadata": metadata,
                "similarity": float(row[2]),
                "created_at": row[3].isoformat() if row[3] else None,
                "tags": row[4],
                "valid_from": row[5].isoformat() if row[5] else None,
                "valid_until": row[6].isoformat() if row[6] else None,
                "event_time": row[7].isoformat() if row[7] else None
            })
            
        return memories

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
