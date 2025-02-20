from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import json
from sklearn.cluster import DBSCAN
import numpy as np
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType
from ..services.llm_service import LLMService
from ..services.embedding_service import EmbeddingService  # Assume this exists

class MemoryDecayHandler:
    def __init__(self,
                 memory_handler: LanceDBMemoryHandler,
                 llm_service: LLMService,
                 embedding_service: EmbeddingService,
                 decay_threshold_days: int = 7,
                 relevance_threshold: float = 0.3,
                 access_threshold: int = 3):
        self.memory_handler = memory_handler
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.decay_threshold_days = decay_threshold_days
        self.relevance_threshold = relevance_threshold
        self.access_threshold = access_threshold

    async def get_memory_clusters(self, memories: List[Dict]) -> List[List[Dict]]:
        """
        Efficient clustering using DBSCAN on embeddings and temporal data
        """
        if not memories:
            return []

        # Get embeddings for all memories
        texts = [memory['content'] for memory in memories]
        embeddings = await self.embedding_service.get_embeddings(texts)
        
        # Normalize timestamps
        timestamps = np.array([m['created_at'].timestamp() for m in memories])
        timestamps_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        
        # Combine embeddings with normalized timestamps
        combined_features = np.column_stack([
            np.array(embeddings),
            timestamps_normalized.reshape(-1, 1) * 0.2  # Weight for temporal aspect
        ])

        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=0.3,  # Distance threshold
            min_samples=2,  # Minimum cluster size
            metric='cosine'
        ).fit(combined_features)

        # Group memories by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(memories[idx])

        return list(clusters.values())

    async def process_with_llm(self, cluster: List[Dict]) -> List[Dict]:
        """
        Process memory cluster with LLM and return list of independent memories
        Returns list of dicts with processed memory data
        """
        existing_tags = await self.memory_handler.get_all_tags()
        
        # Prepare detailed memory context
        memory_contexts = []
        for memory in cluster:
            context = {
                'content': memory['content'],
                'created_at': memory['created_at'].isoformat(),
                'tags': memory.get('tags', []),
                'metadata': memory.get('metadata', {})
            }
            memory_contexts.append(context)

        prompt = f"""
        Analyze these related memories and break them down into independent memory units.
        For each memory unit, provide:

        1. A concise summary
        2. Original content
        3. Relevant tags (from existing: {existing_tags})
        4. Importance score (0-1)
        5. Event time (when the event occurred)
        6. Valid from (when this memory becomes relevant)
        7. Valid until (when this memory stops being relevant)

        Format each memory as markdown with JSON:

        # Memory Unit
        
        Summary: <summary text>
        
        ```json
        {{
            "summary": "detailed summary",
            "content": "original content",
            "tags": ["tag1", "tag2"],
            "importance_score": 0.8,
            "event_time": "2024-02-20T10:00:00Z",
            "valid_from": "2024-02-20T00:00:00Z", 
            "valid_until": "2024-12-31T23:59:59Z"
        }}
        ```

        Memory contexts to analyze:
        {json.dumps(memory_contexts, indent=2)}
        """
        
        llm_response = await self.llm_service.generate(prompt)
        
        # Extract JSON from markdown response
        code_extractor = CodeSnippetExtractor()
        snippets = code_extractor.extract_snippets(llm_response)
        
        # Parse each JSON snippet into a memory dict
        processed_memories = []
        for snippet in snippets:
            if snippet.language == 'json':
                try:
                    memory_data = json.loads(snippet.content)
                    processed_memories.append(memory_data)
                except json.JSONDecodeError:
                    continue
                    
        return processed_memories

    async def resolve_conflicts(self, new_memory: Dict, existing_memory: Dict) -> Dict:
        """Use LLM to intelligently resolve conflicts between memories"""
        prompt = f"""
        Analyze these two memories and provide a merged version that preserves all important information.
        Resolve any conflicts and provide reasoning.

        New Memory:
        {json.dumps(new_memory, indent=2)}

        Existing Memory:
        {json.dumps(existing_memory, indent=2)}

        Response format:
        {{
            "merged_content": "consolidated content",
            "merged_tags": ["tag1", "tag2"],
            "merged_metadata": {{
                "importance_score": 0.8,
                "reasoning": "explanation of merge decisions",
                "preserved_elements": ["element1", "element2"]
            }}
        }}
        """

        llm_response = await self.llm_service.generate(prompt)
        return json.loads(llm_response)

    async def decay_short_term_memories(self):
        """Enhanced decay process with improved clustering and LLM integration"""
        # 1. Selection Process
        short_term_memories = await self.memory_handler.search_similar(
            query="",
            limit=100,
            filter_dict=f"memory_type = '{MemoryType.SHORT_TERM.value}'",
            sort_by="created_at",
            ascending=True
        )

        if not short_term_memories:
            return

        decay_threshold = datetime.now() - timedelta(days=self.decay_threshold_days)
        
        memories_to_decay = [
            memory for memory in short_term_memories
            if (memory['created_at'] < decay_threshold or
                memory.get('access_count', 0) < self.access_threshold)
        ]

        # 2. Enhanced Memory Clustering
        memory_clusters = await self.get_memory_clusters(memories_to_decay)

        # 3 & 4. Enhanced LLM Processing and Integration
        for cluster in memory_clusters:
            try:
                processed_memories = await self.process_with_llm(cluster)
                
                for memory in processed_memories:
                    new_memory_data = {
                        'content': memory['content'],
                        'memory_type': MemoryType.LONG_TERM.value,
                        'metadata': {
                            'summary': memory['summary'],
                            'importance_score': memory['importance_score'],
                            'original_memories': [m['id'] for m in cluster],
                            'consolidated_at': datetime.now().isoformat(),
                            'event_time': memory['event_time'],
                            'valid_from': memory['valid_from'],
                            'valid_until': memory['valid_until']
                        },
                        'tags': memory['tags']
                    }

                # Check for similar existing memories
                existing_similar = await self.memory_handler.search_similar(
                    query=llm_result['summary'],
                    limit=1,
                    filter_dict=f"memory_type = '{MemoryType.LONG_TERM.value}'",
                    threshold=0.9
                )

                if existing_similar:
                    # Use LLM to resolve conflicts and merge memories
                    merged_result = await self.resolve_conflicts(
                        new_memory_data, 
                        existing_similar[0]
                    )

                    await self.memory_handler.update_memory(
                        existing_similar[0]['id'],
                        {
                            'content': merged_result['merged_content'],
                            'tags': merged_result['merged_tags'],
                            'metadata': merged_result['merged_metadata']
                        }
                    )
                else:
                    await self.memory_handler.add_memory(new_memory_data)

                # Delete processed short-term memories
                for memory in cluster:
                    await self.memory_handler.delete_memory(memory['id'])

            except Exception as e:
                print(f"Error processing cluster: {e}")
                continue
