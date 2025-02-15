from typing import List, Dict, Any, Optional
import json
import lancedb
import numpy as np
from datetime import datetime
from pathlib import Path
import uuid
import pandas as pd
import pyarrow as pa

from ..services.llm import LLMService
from ..services.embedding import EmbeddingService
from ..services.code_snippet_extractor import CodeSnippetExtractor

class LanceDBMemoryHandler:
    """Handler for processing and storing memories using LanceDB"""
    
    def __init__(self, 
                 llm_service: LLMService,
                 embedding_service: EmbeddingService,
                 db_path: str = "~/.pyasco/memories"):
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.code_extractor = CodeSnippetExtractor()
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to LanceDB
        self.db = lancedb.connect(str(self.db_path))
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database table with vector search support"""
        schema = pa.schema([
            ("id", pa.string()),
            ("content", pa.string()),
            ("embedding", pa.list_(pa.float32(), 1024)),
            ("memory_type", pa.string()),
            ("metadata", pa.string()),  # JSON string
            ("tags", pa.list_(pa.string())),
            ("created_at", pa.timestamp('us')),
            ("valid_from", pa.timestamp('us')),
            ("valid_until", pa.timestamp('us')),
            ("event_time", pa.timestamp('us'))
        ])
        
        if "memories" not in self.db.table_names():
            self.db.create_table("memories", schema=schema, mode="create")

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
            
            # Prepare the memory data
            memory_data = {
                'id': memory_id,
                'content': memory["content"],
                'embedding': embedding[0].tolist(),
                'memory_type': memory.get("type", "observation"),
                'metadata': json.dumps(metadata),
                'tags': tags,
                'valid_from': valid_from,
                'valid_until': valid_until,
                'event_time': event_time,
                'created_at': datetime.now()
            }

            table = self.db.open_table("memories")
            
            # Check if this is an update to an existing memory
            if memory.get("id"):
                table.delete(f"id = '{memory_id}'")
            
            # Insert the new or updated memory
            table.add([memory_data])
            
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
        
        table = self.db.open_table("memories")
        current_time = datetime.now()
        
        # Perform vector similarity search
        results = table.search(query_embedding).metric("cosine").limit(limit).to_df()
        
        memories = []
        import pdb; pdb.set_trace()
        for _, row in results.iterrows():
            if row._distance > (1 - similarity_threshold):  # Convert cosine similarity to distance
                continue
                
            # Check temporal validity
            valid = True
            if pd.notna(row.valid_from) and pd.notna(row.valid_until):
                valid = row.valid_from <= current_time <= row.valid_until
            elif pd.notna(row.valid_from):
                valid = row.valid_from <= current_time
            elif pd.notna(row.valid_until):
                valid = current_time <= row.valid_until
                
            if not valid:
                continue
                
            # Update access count and last_accessed time
            metadata = json.loads(row.metadata)
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            metadata["last_accessed"] = current_time.isoformat()
            
            # Update the metadata in the database
            table.delete(f"id = '{row.id}'")
            
            # Convert timestamps to pandas Timestamp objects
            valid_from = pd.Timestamp(row.valid_from) if pd.notna(row.valid_from) else None
            valid_until = pd.Timestamp(row.valid_until) if pd.notna(row.valid_until) else None
            event_time = pd.Timestamp(row.event_time) if pd.notna(row.event_time) else None
            created_at = pd.Timestamp(row.created_at) if pd.notna(row.created_at) else pd.Timestamp.now()
            
            table.add([{
                'id': row.id,
                'content': row.content,
                'embedding': row.embedding,
                'memory_type': row.memory_type,
                'metadata': json.dumps(metadata),
                'tags': row.tags,
                'valid_from': valid_from,
                'valid_until': valid_until,
                'event_time': event_time,
                'created_at': created_at
            }])
            
            memories.append({
                "id": str(row.id),
                "content": row.content,
                "metadata": metadata,
                "similarity": 1 - float(row._distance),  # Convert distance back to similarity
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "tags": row.tags,
                "valid_from": row.valid_from.isoformat() if row.valid_from else None,
                "valid_until": row.valid_until.isoformat() if row.valid_until else None,
                "event_time": row.event_time.isoformat() if row.event_time else None,
                "embedding": row.embedding
            })
            
        return memories

    def __del__(self):
        """Cleanup database connection"""
        # LanceDB handles connection cleanup automatically
        pass
