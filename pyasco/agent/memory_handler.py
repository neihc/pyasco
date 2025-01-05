from typing import Dict, Any, List, Optional
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
        self.logger = setup_logger('memory_handler')
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
                        node_spec["properties"].update(context)
                
                # Check if vector index exists for this node label
                if not self._check_vector_index(node_spec["label"]):
                    self.logger.info(f"Creating vector index for label {node_spec['label']}")
                    self.graph_db.execute_query(f"""
                    CREATE VECTOR INDEX {node_spec['label'].lower()}_embeddings IF NOT EXISTS 
                    FOR (n:{node_spec['label']}) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: 1024,
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """)

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
        nodes_info = "\n".join([f"Node {i}: {node}" for i, node in enumerate(nodes)])
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

    def _enhance_search_query(self, query_text: str) -> List[str]:
        """Use LLM to generate multiple enhanced search queries for better semantic matching"""
        prompt = f"""
        Generate 3-5 different search queries based on this original query:
        "{query_text}"

        Consider:
        1. Key concepts and their synonyms
        2. Related technical terms
        3. Broader context that might be relevant
        4. Different aspects or perspectives of the query

        Return only the queries, one per line, no explanations or numbering.
        Each query should be a complete, natural sentence.
        """
        try:
            response = self.llm_service.get_response([{
                "role": "user",
                "content": prompt
            }])
            # Split response into individual queries and clean them
            queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
            # Return original query plus enhanced queries
            return [query_text] + queries
        except Exception as e:
            self.logger.warning(f"Failed to enhance query: {str(e)}")
            return [query_text]

    def _should_explore_node(self, node: Dict, original_query: str, path_so_far: List[Dict]) -> bool:
        """Ask LLM if we should explore this node's neighbors"""
        # Filter out embedding from properties
        properties = {k: v for k, v in node.get('properties', {}).items() if k != 'embedding'}
        node_summary = f"Labels: {node.get('labels', [])}, Properties: {properties}"
        path_summary = "\n".join([
            f"- {p.get('labels', [])} -> {dict((k, v) for k, v in p.get('properties', {}).items() if k != 'embedding')}"
            for p in path_so_far[-3:]  # Show last 3 nodes in path
        ])
        
        prompt = f"""
        Should we explore the neighbors of this node?

        Original query: "{original_query}"
        Current node: {node_summary}
        Path so far: 
        {path_summary}

        Consider:
        1. Is this node relevant to the query?
        2. Would its neighbors likely contain useful information?
        3. Have we already found enough context?
        4. Is the path getting too long or diverging?

        Return only "yes" or "no".
        """
        
        try:
            response = self.llm_service.get_response([{
                "role": "user",
                "content": prompt
            }]).strip().lower()
            return response == "yes"
        except Exception as e:
            self.logger.warning(f"Failed to evaluate node exploration: {str(e)}")
            return False

    def _determine_search_strategy(self, query_text: str) -> Dict[str, Any]:
        """Determine which search strategy to use based on the query"""
        prompt = f"""
        Analyze this query and determine the best search strategy:
        "{query_text}"

        Available strategies:
        1. "vector" - Uses embedding similarity to find semantically similar content
        2. "schema" - Uses graph structure and relationships to find connected information
        3. "both" - Combines results from both strategies

        Consider:
        - Is the query looking for specific facts or relationships?
        - Does it need semantic understanding or exact matches?
        - Would exploring connections be valuable?

        Return a JSON object in a code block:
        ```json
        {{
            "strategy": "vector|schema|both",
            "reason": "brief explanation of choice"
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
                raise ValueError("No strategy recommendation found")
                
            return eval(snippets[0].content)
            
        except Exception as e:
            self.logger.warning(f"Failed to determine strategy: {str(e)}")
            return {"strategy": "vector", "reason": "defaulting to vector search"}

    def recall(self, query_text: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on a natural language query using the most appropriate strategy
        Args:
            query_text (str): Natural language query
            similarity_threshold (float): Minimum similarity score for vector search results
        Returns:
            list: List of relevant memory nodes and their properties
        """
        strategy_info = self._determine_search_strategy(query_text)
        self.logger.info(f"Using search strategy: {strategy_info['strategy']} - {strategy_info['reason']}")
        
        if strategy_info['strategy'] == "schema":
            return self._recall_schema_based(query_text)
        elif strategy_info['strategy'] == "both":
            # Combine results from both strategies
            vector_results = self._recall_vector_based(query_text, similarity_threshold)
            schema_results = self._recall_schema_based(query_text)
            
            # Merge results, avoiding duplicates
            seen_ids = set()
            combined_results = []
            
            for result in vector_results + schema_results:
                node_id = result['n'].element_id
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    combined_results.append(result)
            
            return combined_results
        else:  # Default to vector strategy
            return self._recall_vector_based(query_text, similarity_threshold)
            
    def _recall_schema_based(self, query_text: str, max_attempts: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve memories using schema-based Cypher query generation with iterative refinement
        """
        db_schema = self._get_db_schema()
        attempt = 0
        conversation_history = []
        results = []
        
        try:
            while attempt < max_attempts:
                attempt += 1
                self.logger.info(f"Schema-based recall attempt {attempt}/{max_attempts}")
                
                # Build prompt with conversation history for context
                history_context = "\n\n".join([
                    f"Previous attempt {i+1}:\n{msg['content']}\n"
                    for i, msg in enumerate(conversation_history)
                ])
                
                history_section = f'Previous attempts and errors:\n{history_context}' if history_context else ''
                prompt = f"""
                Given this database schema and natural language query, create a Cypher query.
                
                Query: "{query_text}"
                
                {db_schema}
                
                {history_section}
                
                Requirements:
                1. Use only node labels and relationship types that exist in the schema
                2. Include relevant property filters based on the query
                3. Use appropriate pattern matching and WHERE clauses
                4. Return nodes and relationships that best match the query intent
                5. Limit results to most relevant matches (use LIMIT)
                6. Consider using multiple paths if needed
                7. IMPORTANT: When using ORDER BY, only reference variables that are in scope
                8. If ordering by timestamp, make sure to include it in the WITH/RETURN clause
                9. Always alias complex property references in WITH clauses before using them in ORDER BY
                
                Example structure:
                ```cypher
                MATCH (n:Label)
                WITH n, n.timestamp as timestamp
                ORDER BY timestamp DESC
                RETURN n
                LIMIT 5
                ```
                
                Return only the Cypher query in a code block, nothing else.
                """
                
                response = self.llm_service.get_response([{
                    "role": "user",
                    "content": prompt
                }])
                
                snippets = self.code_extractor.extract_snippets(response)
                if not snippets or not snippets[0].content:
                    error_msg = "No Cypher query generated"
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"Error: {error_msg}"
                    })
                    if attempt == max_attempts:
                        raise ValueError(error_msg)
                    continue
                
                cypher_query = snippets[0].content.strip()
                self.logger.info(f"Generated Cypher query (attempt {attempt}): {cypher_query}")
                
                try:
                    # Execute the generated query
                    results = self.graph_db.execute_query(cypher_query)
                    # If we get here, query executed successfully
                    break
                
                except Exception as e:
                    error_msg = str(e)
                    self.logger.warning(f"Query execution failed (attempt {attempt}): {error_msg}")
                
                    # Add error feedback to conversation history
                    conversation_history.append({
                        "role": "assistant",
                        "content": f"Generated query:\n```cypher\n{cypher_query}\n```\n\nError: {error_msg}\n\nPlease fix the query considering the schema constraints and error message."
                    })
                
                    if attempt == max_attempts:
                        self.logger.error(f"Failed to generate valid query after {max_attempts} attempts. Last error: {error_msg}")
                        return []  # Return empty list instead of raising error
                    continue
            
            # Format results if we have any
            if results:
                formatted_results = []
                for result in results:
                    # Extract node information from each result
                    for key, value in result.items():
                        if hasattr(value, 'labels'):  # It's a node
                            formatted_results.append({
                                'n': value,
                                'score': 1.0,  # Default score for schema-based results
                                'match_type': 'schema',
                                'labels': list(value.labels),
                                'properties': dict(value)
                            })
                return formatted_results
            return []
            
        except Exception as e:
            self.logger.error(f"Error during schema-based recall: {str(e)}")
            return []  # Return empty list for any unexpected errors
            
    def _recall_vector_based(self, query_text: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve memories using vector similarity search
        """
        try:
            # Generate multiple enhanced queries
            enhanced_queries = self._enhance_search_query(query_text)
            self.logger.info(f"Enhanced queries: {enhanced_queries}")
            
            # Search with each query and combine results
            vector_results = []
            seen_node_ids = set()
            
            for enhanced_query in enhanced_queries:
                # Generate embedding for each query
                query_embedding = self.embedding_service.get_embedding(enhanced_query).tolist()[0]
            
                # Search with current query embedding
                current_results = []
                labels_with_indexes = self.graph_db.execute_query("""
            SHOW INDEXES
            YIELD name, type, labelsOrTypes
            WHERE type = 'VECTOR'
            RETURN distinct labelsOrTypes[0] as label
            """)
            
            # Query each indexed label
            for label_result in labels_with_indexes:
                label = label_result['label']
                label_results = self.graph_db.execute_query(f"""
                CALL db.index.vector.queryNodes($index_name, $k, $query)
                YIELD node, score 
                WHERE score >= $threshold
                RETURN node, score
                ORDER BY score DESC
                """, {
                    "index_name": f"{label.lower()}_embeddings",
                    "k": 5,  # Reduced initial results per label
                    "query": query_embedding,
                    "threshold": similarity_threshold
                })
                # Only add results for nodes we haven't seen yet
                for result in label_results:
                    node_id = result['node'].element_id
                    if node_id not in seen_node_ids:
                        seen_node_ids.add(node_id)
                        current_results.append(result)
                
                vector_results.extend(current_results)
            
            if not vector_results:
                self.logger.info("No similar nodes found via vector search")
                return []
            
            self.logger.info(f"Found {len(vector_results)} total results across {len(enhanced_queries)} queries")

            # Sort and prepare for neighborhood exploration
            vector_results.sort(key=lambda x: x['score'], reverse=True)
            seen_ids = set()
            final_results = []
            nodes_to_explore = []

            # Add initial vector results
            for result in vector_results:
                node = result['node']
                node_id = node.element_id
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    node_info = {
                        'n': node,
                        'score': result['score'],
                        'match_type': 'vector',
                        'labels': list(node.labels),
                        'properties': dict(node)
                    }
                    final_results.append(node_info)
                    nodes_to_explore.append({
                        'node_id': node_id,
                        'labels': list(node.labels),
                        'properties': dict(node),
                        'path': [node_info]
                    })

            # Explore neighborhoods of similar nodes
            max_depth = 3
            for start_node in nodes_to_explore:
                current_depth = 0
                nodes_at_depth = [start_node]
                
                while current_depth < max_depth and nodes_at_depth:
                    next_level = []
                    for current in nodes_at_depth:
                        # Get all neighbors
                        neighbors = self.graph_db.execute_query("""
                        MATCH (n)-[r]-(neighbor)
                        WHERE elementId(n) = $node_id
                        RETURN neighbor, type(r) as relationship_type
                        """, {"node_id": current['node_id']})
                        
                        for neighbor in neighbors:
                            neighbor_node = neighbor['neighbor']
                            neighbor_id = neighbor_node.element_id
                            
                            if neighbor_id in seen_ids:
                                continue
                                
                            neighbor_info = {
                                'node_id': neighbor_id,
                                'labels': list(neighbor_node.labels),
                                'properties': dict(neighbor_node),
                                'path': current['path'] + [{
                                    'n': neighbor_node,
                                    'score': 0.5,  # Base score for neighbors
                                    'match_type': 'neighbor',
                                    'relationship': neighbor['relationship_type']
                                }]
                            }
                            
                            # Ask LLM if we should explore this neighbor
                            if self._should_explore_node(
                                neighbor_info,
                                query_text,
                                current['path']
                            ):
                                seen_ids.add(neighbor_id)
                                # Properly format neighbor node info
                                final_results.append({
                                    'n': neighbor_node,
                                    'score': 0.5,  # Base score for neighbors
                                    'match_type': 'neighbor',
                                    'relationship': neighbor['relationship_type'],
                                    'labels': list(neighbor_node.labels),
                                    'properties': dict(neighbor_node)
                                })
                                next_level.append(neighbor_info)
                    
                    nodes_at_depth = next_level
                    current_depth += 1

            self.logger.info(f"Found {len(final_results)} total results through neighborhood exploration")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error during recall: {str(e)}")
            raise

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

    def _check_vector_index(self, node_label: str) -> bool:
        """Check if vector index exists for a given node label"""
        try:
            result = self.graph_db.execute_query("""
            SHOW INDEXES
            YIELD name, type, labelsOrTypes, properties
            WHERE type = 'VECTOR' 
            AND $label IN labelsOrTypes
            AND 'embedding' IN properties
            RETURN count(*) as count
            """, {"label": node_label})
            
            return result[0]['count'] > 0
        except Exception as e:
            self.logger.error(f"Failed to check vector index: {str(e)}")
            return False
