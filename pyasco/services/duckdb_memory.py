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
                memory_type TEXT NOT NULL DEFAULT 'observation',
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

    def remember(self, content: str, context: Optional[Dict[str, Any]] = None, 
                related_memories: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Extract meaningful memories from content and store them in the database
        
        Args:
            content: The text content to extract memories from
            context: Optional contextual metadata
            related_memories: Optional list of related existing memories to consider for updates
            
        Returns:
            Dict containing status and extracted memories
        """
        # Use LLM to extract meaningful memories
        related_memories_text = ""
        if related_memories:
            related_memories_text = "RELATED MEMORIES:\n" + "\n".join(
                f"ID: {mem.get('id', 'unknown')}\nContent: {mem.get('content', '')}\n"
                for mem in related_memories
            )

        current_time = datetime.now().isoformat()
        prompt = f"""
        Extract and update memories from the following content, considering any related existing memories.
        Focus on distinct, actionable information and avoid generic or abstract concepts.
        Each memory should capture a single, well-defined piece of information.

        Current datetime: {current_time}

        Guidelines:
        - Include specific details, numbers, dates, names, or actions
        - Break down complex information into individual memories
        - Exclude vague or general statements
        - Focus on factual, verifiable information
        - Identify temporal aspects (when the information is valid or occurred)
        - Assign relevant categorical tags
        - For existing memories: update, refine, or merge if new information is relevant
        - Mark conflicts or contradictions with existing memories

        CONTENT:
        {content}
        
        CONTEXT:
        {context or {}}

        {related_memories_text}
        
        Return a JSON object wrapped in a code block with this structure:
        ```json
        {{
            "memories": [
                {{
                    "id": "existing-id-to-update or null for new memory",
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
                        "access_count": 0,
                        "update_type": "new|update|merge|conflict"
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
                "confidence": memory.get("confidence", 1.0)
            }
            
            # Add any additional context
            if context:
                metadata.update(context)
            
            # Use existing ID or generate new one
            memory_id = memory.get("id") or str(uuid.uuid4())
            # Extract temporal data
            temporal = memory.get("temporal", {})
            valid_from = temporal.get("valid_from")
            valid_until = temporal.get("valid_until")
            event_time = temporal.get("event_time")
            
            # Extract tags
            tags = memory.get("tags", [])
            
            # Check if this is an update to an existing memory
            if memory.get("id"):
                self.conn.execute("""
                    UPDATE memories 
                    SET content = $content,
                        embedding = $embedding::FLOAT[],
                        memory_type = $memory_type,
                        metadata = $metadata,
                        tags = $tags::TEXT[],
                        valid_from = $valid_from::TIMESTAMP,
                        valid_until = $valid_until::TIMESTAMP,
                        event_time = $event_time::TIMESTAMP
                    WHERE id = $id;
                """, {
                    'content': memory["content"],
                    'embedding': embedding[0].tolist(),
                    'memory_type': memory.get("type", "observation"),
                    'metadata': json.dumps(metadata),
                    'tags': tags,
                    'valid_from': valid_from,
                    'valid_until': valid_until,
                    'event_time': event_time,
                    'id': memory_id
                })
            else:
                self.conn.execute("""
                    INSERT INTO memories (
                        id, content, embedding, memory_type, metadata, tags,
                        valid_from, valid_until, event_time
                    )
                    VALUES ($id, $content, $embedding::FLOAT[], $memory_type, $metadata, $tags::TEXT[],
                            $valid_from::TIMESTAMP, $valid_until::TIMESTAMP, $event_time::TIMESTAMP);
                """, {
                    'id': memory_id,
                    'content': memory["content"],
                    'embedding': embedding[0].tolist(),
                    'memory_type': memory.get("type", "observation"),
                    'metadata': json.dumps(metadata),
                    'tags': tags,
                    'valid_from': valid_from,
                    'valid_until': valid_until,
                    'event_time': event_time
                })
            
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
        
        query = """
            SELECT 
                id,
                content,
                metadata,
                array_cosine_similarity(embedding, $query_embedding::FLOAT[1024]) as similarity,
                created_at,
                tags,
                valid_from,
                valid_until,
                event_time,
                embedding
            FROM memories
            WHERE array_cosine_similarity(embedding, $query_embedding::FLOAT[1024]) >= $similarity_threshold
            AND (
                (valid_from IS NULL AND valid_until IS NULL) OR
                (valid_from IS NULL AND valid_until > CURRENT_TIMESTAMP) OR
                (valid_from <= CURRENT_TIMESTAMP AND valid_until IS NULL) OR
                (valid_from <= CURRENT_TIMESTAMP AND valid_until > CURRENT_TIMESTAMP)
            )
            ORDER BY array_cosine_similarity(embedding, $query_embedding::FLOAT[1024]) DESC
            LIMIT $limit;
        """
        params = {
            'query_embedding': query_embedding,
            'similarity_threshold': similarity_threshold,
            'limit': limit
        }
        results = self.conn.execute(query, params).fetchall()
        
        memories = []
        for row in results:
            # Update access count and last_accessed time
            metadata = json.loads(row[2])
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            metadata["last_accessed"] = datetime.now().isoformat()
            
            # Update the metadata in the database
            self.conn.execute("""
                UPDATE memories 
                SET metadata = $metadata 
                WHERE id = $id
            """, {
                'metadata': json.dumps(metadata),
                'id': row[0]  # row[0] is the id
            })
            
            memories.append({
                "id": str(row[0]),
                "content": row[1],
                "metadata": metadata,
                "similarity": float(row[3]),
                "created_at": row[4].isoformat() if row[4] else None,
                "tags": row[5],
                "valid_from": row[6].isoformat() if row[6] else None,
                "valid_until": row[7].isoformat() if row[7] else None,
                "event_time": row[8].isoformat() if row[8] else None,
                "embedding": row[9]
            })
            
        return memories

    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
