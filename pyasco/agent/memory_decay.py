import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import json
import copy
import numpy as np

from sklearn.cluster import DBSCAN

from ..services.lance_memory import LanceDBMemoryHandler, MemoryType
from ..services.llm import LLMService
from ..services.code_snippet_extractor import CodeSnippetExtractor

class MemoryDecayHandler:
    def _prepare_memory_for_llm(self, memory: Dict) -> Dict:
        """Prepare memory dict for LLM by removing vector and making JSON-safe"""
        memory_copy = copy.deepcopy(memory)
        
        # Remove vector field which isn't JSON serializable
        memory_copy.pop('vector', None)
        memory_copy.pop('_relevance_score', None)
        
        # Convert numpy arrays to lists
        if 'tags' in memory_copy and isinstance(memory_copy['tags'], np.ndarray):
            memory_copy['tags'] = memory_copy['tags'].tolist()
            
        # Convert datetime objects and handle NaT values
        for field in ['created_at', 'valid_from', 'valid_until', 'event_time']:
            if field in memory_copy:
                if isinstance(memory_copy[field], datetime):
                    memory_copy[field] = memory_copy[field].isoformat()
                elif isinstance(memory_copy[field], str) and memory_copy[field] == 'NaT':
                    memory_copy[field] = None
                    
        # Convert metadata to proper dict if it's a string
        if isinstance(memory_copy.get('metadata'), str):
            try:
                memory_copy['metadata'] = json.loads(memory_copy['metadata'])
            except json.JSONDecodeError:
                memory_copy['metadata'] = {}
                
        return memory_copy

    def __init__(self,
                 memory_handler: LanceDBMemoryHandler,
                 llm_service: LLMService,
                 decay_threshold_days: int = 1,
                 relevance_threshold: float = 0.3,
                 access_threshold: int = 5):
        self.memory_handler = memory_handler
        self.llm_service = llm_service
        self.decay_threshold_days = decay_threshold_days
        self.relevance_threshold = relevance_threshold
        self.access_threshold = access_threshold

    async def get_memory_clusters(self, memories: List[Dict]) -> List[List[Dict]]:
        """
        Efficient clustering using DBSCAN on embeddings and temporal data
        """
        if not memories:
            return []

        # Use existing embeddings from memories
        embeddings = np.array([memory['vector'] for memory in memories])
        
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
        memory_contexts = [self._prepare_memory_for_llm(memory) for memory in cluster]

        prompt = f"""
        Analyze these related memories and break them down into independent memory units.
        For each memory unit, provide:

        1. A concise summary
        2. Original content
        3. Relevant tags (from existing: {existing_tags})
        4. Importance score (0-1) based on these criteria:
           - 0.8-1.0: Critical information (core concepts, key decisions, major events)
           - 0.6-0.8: Important details (specific examples, implementation details)
           - 0.4-0.6: Supporting information (context, background, minor details)
           - 0.2-0.4: Supplementary details (temporary notes, partial information)
           - 0.0-0.2: Trivial information (redundant or obsolete details)
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
        
        llm_response = self.llm_service.get_response([{"role": "user", "content": prompt}])
        
        # Extract JSON from markdown response
        code_extractor = CodeSnippetExtractor()
        snippets = code_extractor.extract_snippets(llm_response)
        
        # Parse each JSON snippet into a memory dict
        processed_memories = []
        for snippet in snippets:
            if snippet.language == 'json':
                try:
                    data = json.loads(snippet.content)
                    memory_data = {
                        'id': str(uuid.uuid4()),
                        'content': data['content'],
                        'memory_type': MemoryType.LONG_TERM.value,
                        'metadata': json.dumps({
                            'event_time': data['event_time'],
                            'valid_from': data['valid_from'],
                            'valid_until': data['valid_until'],
                            'consolidated_at': datetime.now().isoformat()
                        }),
                        'tags': data['tags'],
                        'created_at': datetime.now(),
                        'valid_from': datetime.fromisoformat(data['valid_from']),
                        'valid_until': datetime.fromisoformat(data['valid_until']),
                        'event_time': datetime.fromisoformat(data['event_time']),
                        'access_count': 0,
                        'importance_score': data['importance_score'],
                        'summary': data['summary']
                    }
                    processed_memories.append(memory_data)
                except (json.JSONDecodeError, KeyError):
                    continue
                    
        return processed_memories

    async def should_integrate_memories(self, new_memory: Dict, existing_memory: Dict) -> Tuple[bool, Dict]:
        """Use LLM to decide if and how to integrate memories"""
        prompt = f"""
        Analyze these two memories and decide if they should be integrated.
        Consider:
        1. Are they truly about the same topic/event?
        2. Would combining them preserve more useful information?
        3. How should they be merged if integration is recommended?
        4. Calculate importance score (0-1) for integrated memory based on:
           - 0.8-1.0: Critical information (core concepts, key decisions, major events)
           - 0.6-0.8: Important details (specific examples, implementation details)
           - 0.4-0.6: Supporting information (context, background, minor details)
           - 0.2-0.4: Supplementary details (temporary notes, partial information)
           - 0.0-0.2: Trivial information (redundant or obsolete details)
           The integrated memory should have an importance score >= max(existing_score, new_score)
           if it contains more complete/valuable information.

        New Memory:
        {json.dumps(self._prepare_memory_for_llm(new_memory), indent=2)}

        Existing Memory:
        {json.dumps(self._prepare_memory_for_llm(existing_memory), indent=2)}

        Response format:
        ```json
        {{
            "should_integrate": true/false,
            "reasoning": "detailed explanation of the decision",
            "integrated_memory": {{  \# Only if should_integrate is true
                "content": "merged content",
                "summary": "updated summary",
                "tags": ["tag1", "tag2"],
                "importance_score": 0.8,
                "valid_from": "2024-02-20T00:00:00Z",
                "valid_until": "2024-12-31T23:59:59Z",
                "event_time": "2024-02-20T10:00:00Z"
            }}
        }}
        ```
        """

        llm_response = self.llm_service.get_response([{"role": "user", "content": prompt}])
        
        # Extract JSON using CodeSnippetExtractor
        code_extractor = CodeSnippetExtractor()
        snippets = code_extractor.extract_snippets(llm_response)
        
        # Find the JSON snippet
        for snippet in snippets:
            if snippet.language == 'json':
                try:
                    result = json.loads(snippet.content)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            return False, None
            
        if result.get("should_integrate"):
            integrated = result.get("integrated_memory", {})
            # Preserve the existing memory's ID and metadata structure
            memory_data = {
                'id': existing_memory['id'],
                'content': integrated['content'],
                'memory_type': MemoryType.LONG_TERM.value,
                'metadata': {
                    'intergrated_reason': result.get('reasoning', ''),
                    'integrated_at': datetime.now().isoformat()
                },
                'tags': integrated['tags'],
                'created_at': existing_memory['created_at'],
                'valid_from': datetime.fromisoformat(integrated['valid_from']),
                'valid_until': datetime.fromisoformat(integrated['valid_until']),
                'event_time': datetime.fromisoformat(integrated['event_time']),
                'access_count': existing_memory['access_count'],
                'importance_score': integrated['importance_score'],
                'summary': integrated['summary']
            }
            return True, memory_data
        
        return False, None

    async def decay_short_term_memories(self):
        """Enhanced decay process with improved clustering and LLM integration"""
        # 1. Selection Process - Using SQL query
        query = f"""
        SELECT *
        FROM memories 
        WHERE memory_type = '{MemoryType.SHORT_TERM.value}'
        ORDER BY created_at ASC
        LIMIT 100
        """
        
        short_term_memories = await self.memory_handler.sql_query(query)
        
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
                        'id': str(uuid.uuid4()),
                        'content': memory['content'],
                        'memory_type': MemoryType.LONG_TERM.value,
                        'metadata': json.dumps({
                            'original_memories': [m['id'] for m in cluster],
                            'consolidated_at': datetime.now().isoformat()
                        }),
                        'tags': memory['tags'],
                        'created_at': datetime.now(),
                        'valid_from': memory.get('valid_from'),
                        'valid_until': memory.get('valid_until'),
                        'event_time': memory.get('event_time'),
                        'access_count': 0,
                        'importance_score': memory['importance_score'],
                        'summary': memory['summary']
                    }

                    # Check for similar existing memories with lower threshold
                    existing_similar = await self.memory_handler.search_similar(
                        query=memory['summary'],
                        limit=5,  # Get more potential matches
                        filter_dict=f"memory_type = '{MemoryType.LONG_TERM.value}'",
                        score_threshold=0.3  # Lower threshold to catch more potential matches
                    )

                    integrated = False
                    for existing in existing_similar:
                        should_integrate, integrated_memory = await self.should_integrate_memories(
                            new_memory_data,
                            existing
                        )
                        
                        if should_integrate:
                            await self.memory_handler.update_memory(
                                existing['id'],
                                integrated_memory
                            )
                            integrated = True
                            break
                    
                    if not integrated:
                        await self.memory_handler.add_memory(new_memory_data)

                # Delete processed short-term memories
                for memory in cluster:
                    await self.memory_handler.delete_memory(memory['id'])

            except Exception as e:
                print(f"Error processing cluster: {e}")
                continue
