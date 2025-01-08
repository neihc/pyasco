from typing import Dict, Any, List, Optional, Tuple
from ..services.graphdb import GraphDB
from ..services.llm import LLMService
from ..services.code_snippet_extractor import CodeSnippetExtractor
from ..services.embedding import EmbeddingService
from ..logger_config import setup_logger


class MemoryHandler:
    """Handler for processing and storing memories using LLM and graph database"""
    
    def __init__(self, memory_instructions: str, llm_service: Optional[LLMService] = None,
                 graph_db: Optional[GraphDB] = None, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize the memory handler
        Args:
            memory_instructions (str): Additional instructions for how to process and structure memories
            llm_service (LLMService): LLM service instance for processing memories
            graph_db (GraphDB): GraphDB instance for storing memories
        """
        self.logger = setup_logger('memory_handler', log_file='memory_handler.log', verbose=True)
        self.memory_instructions = memory_instructions
        self.llm_service = llm_service or LLMService()
        self.graph_db = graph_db or GraphDB()
        self.code_extractor = CodeSnippetExtractor()
        self.embedding_service = embedding_service or EmbeddingService()

    def _generate_node_embedding(self, node_data: Dict[str, Any]) -> str:
        """
        Generate embedding for node data
        Args:
            node_data (dict): Node properties to embed
        Returns:
            str: Flattened text representation of node data
        """
        # Flatten node data into a string representation
        flat_text = " ".join([
            f"{key}: {str(value)}" 
            for key, value in node_data.items() 
            if isinstance(value, (str, int, float, bool))
        ])
        
        # Generate embedding
        embedding = self.embedding_service.get_embedding(flat_text)
        return embedding.tolist()[0]  # Convert numpy array to list

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Recursively flatten a nested dictionary with key prefixing"""
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to strings to avoid Neo4j array type issues
                items.append((new_key, str(v)))
            else:
                # Convert all values to strings to ensure primitive types
                items.append((new_key, str(v)))
        return dict(items)

    def _create_nodes(self, content: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        First phase: Create nodes based on content
        Args:
            content (str): The content to be remembered
            context (dict): Optional contextual information
        Returns:
            list: List of created node information
        """
        prompt = f"""
        Process the following content into node structures.
        
        CONTENT:
        {content}
        
        CONTEXT:
        {context or {}}
        
        INSTRUCTIONS:
        {self.memory_instructions}
        
        Based on the provided schema in the instructions, structure this content as nodes only.
        Return a JSON object wrapped in a code block with only the nodes array:
        Only response string value, do not response null/numberic
        ```json
        {{
            "nodes": [
                {{
                    "label": "node type from schema",
                    "properties": {{
                        "property1": "value1",
                        ...
                    }}
                }}
            ]
        }}
        ```
        """
        
        try:
            response = self.llm_service.get_response([{
                "role": "user",
                "content": prompt
            }])
            
            snippets = self.code_extractor.extract_snippets(response)
            if not snippets or not snippets[0].content:
                raise ValueError("No JSON structure found in LLM response")
                
            node_structure = eval(snippets[0].content)
            
            created_nodes = []
            for node_spec in node_structure["nodes"]:
                if not created_nodes:
                    node_spec["properties"]["original_content"] = content
                    if context:
                        # Recursively flatten any nested objects in context
                        flattened_context = self._flatten_dict(context, prefix='context')
                        node_spec["properties"].update(flattened_context)
                
                # Filter out null/None values and flatten any nested objects in the node properties
                filtered_properties = {k: v for k, v in node_spec["properties"].items() if v is not None and v != "null"}
                node_spec["properties"] = self._flatten_dict(filtered_properties)
                
                # Ensure vector index exists for this node label
                self.graph_db.ensure_vector_index(node_spec["label"])

                # Generate embedding for node properties
                node_spec["properties"]["embedding"] = self._generate_node_embedding(node_spec["properties"])
                
                node = self.graph_db.create_node(
                    node_spec["label"],
                    node_spec["properties"]
                )
                created_nodes.append(node)
            
            return created_nodes
            
        except Exception as e:
            self.logger.error(f"Failed to create nodes: {str(e)}")
            raise
            
    def _create_relationships(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Second phase: Create relationships between nodes
        Args:
            nodes (list): List of created nodes
        Returns:
            list: List of created relationship information
        """
        # Filter out embeddings from node info
        filtered_nodes = []
        for node in nodes:
            # Create a new dict instead of copying the Neo4j Node object
            filtered_node = {
                'labels': list(node.labels) if hasattr(node, 'labels') else [],
                'properties': {k: v for k, v in dict(node).items() if k != 'embedding'},
                'node_id': node.element_id if hasattr(node, 'element_id') else None
            }
            filtered_nodes.append(filtered_node)
            
        nodes_info = "\n".join([f"Node {i}: {node}" for i, node in enumerate(filtered_nodes)])
        prompt = f"""
        Create relationships between the following nodes:
        
        {nodes_info}
        
        INSTRUCTIONS:
        {self.memory_instructions}
        
        Return a JSON object wrapped in a code block with only the relationships array:
        ```json
        {{
            "relationships": [
                {{
                    "from_node_id": "unique_id_1",
                    "to_node_id": "unique_id_2",
                    "type": "RELATIONSHIP_TYPE",
                    "properties": {{
                        "property1": "value1",
                        ...
                    }}
                }}
            ]
        }}
        ```
        """
        
        try:
            response = self.llm_service.get_response([{
                "role": "user",
                "content": prompt
            }])
            
            snippets = self.code_extractor.extract_snippets(response)
            if not snippets or not snippets[0].content:
                raise ValueError("No JSON structure found in LLM response")
                
            rel_structure = eval(snippets[0].content)
            
            created_relationships = []
            for rel in rel_structure["relationships"]:
                relationship = self.graph_db.create_relationship(
                    rel["from_node_id"],
                    rel["to_node_id"],
                    rel["type"],
                    rel.get("properties", {})
                )
                created_relationships.append(relationship)
            
            return created_relationships
            
        except Exception as e:
            self.logger.error(f"Failed to create relationships: {str(e)}")
            raise

    def remember(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process content and store it as a memory in the graph database
        Args:
            content (str): The content to be remembered
            context (dict): Optional contextual information
        Returns:
            dict: The stored memory node properties
        """
        try:
            # Phase 1: Create nodes
            created_nodes = self._create_nodes(content, context)
            self.logger.info(f"Created {len(created_nodes)} nodes")
            
            # Phase 2: Create relationships
            if len(created_nodes) > 1:
                created_relationships = self._create_relationships(created_nodes)
                self.logger.info(f"Created {len(created_relationships)} relationships")
            
            return created_nodes[0]  # Return the primary node
            
        except Exception as e:
            self.logger.error(f"Failed to process memory: {str(e)}")
            raise

    def _search_similar(self, search_text: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar nodes using vector similarity with enhanced queries
        Args:
            search_text (str): Text to search for
            similarity_threshold (float): Minimum similarity score
        Returns:
            list: List of similar nodes with scores
        """
        try:
            # Generate embedding for search text
            query_embedding = self.embedding_service.get_embedding(search_text).tolist()[0]
            
            # Search with the query
            results = []
                
                # Get results from all indexed labels
                for label in self.graph_db.get_indexed_labels():
                    label_results = self.graph_db.get_vector_search_results(
                        label,
                        query_embedding,
                        similarity_threshold
                    )
                    results.extend(label_results)
            
            # Sort by score and format results
            results.sort(key=lambda x: x['score'], reverse=True)
            return [{
                'node': result['node'],
                'score': result['score'],
                'labels': list(result['node'].labels),
                'properties': {k:v for k,v in dict(result['node']).items() if k != 'embedding'}
            } for result in results]
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            return []

    def _execute_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return formatted results
        Args:
            cypher_query (str): Cypher query to execute
        Returns:
            list: Query results formatted as dictionaries
        """
        try:
            results = self.graph_db.execute_query(cypher_query)
            formatted_results = []
            
            for result in results:
                # Extract and format node data from results
                for value in result.values():
                    if hasattr(value, 'labels'):  # It's a node
                        formatted_results.append({
                            'node': value,
                            'labels': list(value.labels),
                            'properties': dict(value)
                        })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            return []

    def recall(self, query_text: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on natural language query using LLM to guide the search
        Args:
            query_text (str): Natural language query
        Returns:
            list: List of relevant memory nodes
        """
        prompt = f"""
        Help find relevant information using these available functions:
        
        1. Search similar nodes:
           - Input: search text
           - Returns: nodes with similarity scores
           - Best for: finding semantically similar content
        
        2. Execute Cypher query:
           - Input: Cypher query following Neo4j syntax
           - Returns: matching nodes
           - Best for: specific patterns, relationships, or conditions
        
        Original query: "{query_text}"
        
        First, generate 3-4 variations of the search query:
        1. Use synonyms and related technical terms
        2. Include domain-specific terminology
        3. Consider both broader and narrower scopes
        4. Rephrase using different sentence structures
        
        Then decide which function(s) to use and in what order.
        
        Database schema:
        {self._get_db_schema()}
        
        Return ONLY code blocks, alternating between:
        
        For similarity search:
        ```text
        variation 1 of search text
        ```
        ```text
        variation 2 of search text
        ```
        
        For Cypher query:
        ```cypher
        your query here
        ```
        
        I will execute each code block and return results.
        You can then analyze them and provide more code blocks if needed.
        """
        
        all_results = []
        seen_node_ids = set()
        max_iterations = 3
        
        for iteration in range(max_iterations):
            # Add context from previous results if any
            if all_results:
                result_summary = "\n".join([
                    f"- Node {r['labels']}: {r['properties'].get('content', '')[:100]}..."
                    for r in all_results[-3:]  # Show last 3 results
                ])
                prompt += f"\n\nPrevious results:\n{result_summary}"
            
            response = self.llm_service.get_response([{
                "role": "user",
                "content": prompt
            }])
            
            # Extract code snippets and execute appropriate function
            snippets = self.code_extractor.extract_snippets(response)
            if not snippets:
                break
                
            for snippet in snippets:
                try:
                    if snippet.language == 'text':
                        results = self._search_similar(snippet.content)
                    elif snippet.language == 'cypher':
                        results = self._execute_query(snippet.content)
                    else:
                        continue
                        
                    # Add new unique results
                    for result in results:
                        node_id = result['node'].element_id
                        if node_id not in seen_node_ids:
                            seen_node_ids.add(node_id)
                            all_results.append(result)
                            
                except Exception as e:
                    self.logger.error(f"Error processing snippet: {str(e)}")
                    continue
            
            # Check if we have enough relevant results
            if len(all_results) >= 5:  # Arbitrary threshold
                break
                
        # Sort results by score and return top 5
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        filtered_results = [r for r in all_results if r.get('score', 0) >= similarity_threshold]
        return filtered_results[:5]
            
    def _get_db_schema(self) -> str:
        """Get the actual schema from the database and combine with domain schema"""
        try:
            # Get schema from GraphDB
            db_schema = self.graph_db.get_schema()
            
            # Format complete schema with domain context
            schema = "Database Schema:\n\n"
            
            # Add actual database state
            schema += "Current Database State:\n\n" + db_schema
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Failed to get database schema: {str(e)}")
            return ""
