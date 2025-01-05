from typing import Dict, Any, List, Optional
from ..services.graphdb import GraphDB
from ..services.llm import LLMService
from ..services.code_snippet_extractor import CodeSnippetExtractor
from ..services.embedding import EmbeddingService
from ..logger_config import setup_logger

DOMAIN_SCHEMA_INSTRUCTIONS = """
Core domain schema for AI agent memory:

User nodes:
- Label: "User"
- Properties:
  - user_id: unique identifier
  - last_interaction: timestamp
  - interaction_count: number of interactions

Conversation nodes:
- Label: "Conversation"
- Properties:
  - conversation_id: unique identifier
  - timestamp: when the conversation occurred
  - summary: brief summary of the conversation
  - content: full conversation content

Message nodes:
- Label: "Message"
- Properties:
  - role: who sent the message (user/assistant)
  - content: message content
  - timestamp: when the message was sent

Relationships:
- USER_PARTICIPATED -> between User and Conversation
- CONTAINS_MESSAGE -> between Conversation and Message
- SENT_MESSAGE -> between User and Message
"""

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
        self.memory_instructions = f"{DOMAIN_SCHEMA_INSTRUCTIONS}\n\nAdditional Schema:\n{memory_instructions}"
        self.llm_service = llm_service or LLMService()
        self.graph_db = graph_db or GraphDB()
        self.code_extractor = CodeSnippetExtractor()
        self.embedding_service = embedding_service or EmbeddingService()
        self._setup_indexes()
        self._setup_vector_indexes()

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
            
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract key entities from the query using LLM
        """
        prompt = f"""
        Extract key entities and concepts from this query:
        "{query}"
        
        Return a JSON array of entities in a code block. Include:
        - Important nouns and noun phrases
        - Technical terms
        - Action verbs
        - Time references
        - Any specific identifiers
        
        Example:
        ```json
        ["python code", "error handling", "last week", "database connection"]
        ```
        """
        
        try:
            response = self.llm_service.get_response([{
                "role": "user", 
                "content": prompt
            }])
            
            snippets = self.code_extractor.extract_snippets(response)
            if not snippets or not snippets[0].content:
                raise ValueError("No entities found in LLM response")
                
            return eval(snippets[0].content)
            
        except Exception as e:
            self.logger.error(f"Failed to extract entities: {str(e)}")
            return []

    def _should_expand_search(self, query: str, current_results: List[Dict]) -> bool:
        """
        Ask LLM if search should be expanded based on current results
        """
        results_summary = "\n".join([
            f"- Node type: {r['n'].labels}, Properties: {dict(r['n'])}, Score: {r['score']}"
            for r in current_results[:3]  # Summarize top 3 results
        ])
        
        prompt = f"""
        Query: "{query}"
        
        Current top results:
        {results_summary}
        
        Should we expand the search to find more related nodes? Consider:
        1. Are the current results directly relevant to the query?
        2. Would exploring connected nodes add valuable context?
        3. Are there missing aspects of the query not covered by current results?
        
        Return only "yes" or "no".
        """
        
        try:
            response = self.llm_service.get_response([{
                "role": "user",
                "content": prompt
            }]).strip().lower()
            
            return response == "yes"
            
        except Exception as e:
            self.logger.error(f"Failed to determine search expansion: {str(e)}")
            return False

    def _search_entity(self, entity: str, node_types: Optional[List[str]], limit: int) -> List[Dict[str, Any]]:
        """
        Search for an entity in the graph database
        """
        label_filter = ""
        if node_types:
            labels_list = [f"'{label}'" for label in node_types]
            label_filter = f"WHERE any(label IN labels(n) WHERE label IN [{', '.join(labels_list)}])"
        
        query = f"""
        MATCH (n)
        {label_filter}
        WITH n, [prop IN keys(n) WHERE n[prop] CONTAINS $entity] AS matches
        WHERE size(matches) > 0
        RETURN n AS node
        LIMIT $limit
        """
        
        results = self.graph_db.execute_query(query, {"entity": entity, "limit": limit})
        return [{'node': result['node']} for result in results]

    def _expand_entity(self, entity: str) -> str:
        """
        Expand the entity for broader search
        """
        # Placeholder for entity expansion logic
        return entity + " expanded"

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

    def _build_query(self, query_text: str) -> str:
        """
        Use LLM to build a Cypher query based on the natural language query
        """
        prompt = f"""
        Convert this natural language query into a Cypher query for Neo4j:
        "{query_text}"

        Use this schema:
        {self.memory_instructions}

        Return only the Cypher query in a code block, nothing else.
        The query should:
        - Use appropriate node labels and relationship types from the schema
        - Include relevant property filters
        - Return nodes and relationships that best match the query intent
        - Limit results to 5 most relevant matches
        """
        
        try:
            response = self.llm_service.get_response([{
                "role": "user",
                "content": prompt
            }])
            
            snippets = self.code_extractor.extract_snippets(response)
            if not snippets or not snippets[0].content:
                raise ValueError("No Cypher query found in LLM response")
                
            return snippets[0].content.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to build query: {str(e)}")
            raise

    def _evaluate_results(self, query_text: str, results: List[Dict]) -> Dict[str, Any]:
        """
        Use LLM to evaluate query results and suggest improvements
        """
        results_summary = "\n".join([
            f"- Node: {r.get('n', {}).get('properties', {})} Score: {r.get('score', 'N/A')}"
            for r in results[:3]
        ])
        
        prompt = f"""
        Evaluate these query results:
        
        Original query: "{query_text}"
        
        Results:
        {results_summary}
        
        Return a JSON object in a code block with this structure:
        ```json
        {{
            "sufficient": true/false,
            "reason": "explanation of why results are sufficient or not",
            "improved_query": "suggested improved cypher query if needed"
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
                raise ValueError("No evaluation found in LLM response")
                
            return eval(snippets[0].content)
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate results: {str(e)}")
            return {"sufficient": True, "reason": "Error in evaluation"}

    def recall(self, query_text: str, max_iterations: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on a natural language query using iterative refinement
        Args:
            query_text (str): Natural language query
            max_iterations (int): Maximum number of query refinement iterations
        Returns:
            list: List of relevant memory nodes and their properties
        """
        iteration = 0
        best_results = []
        
        while iteration < max_iterations:
            try:
                # Build or use existing query
                if iteration == 0:
                    cypher_query = self._build_query(query_text)
                
                # Generate embedding for query
                query_embedding = self.embedding_service.get_embedding(query_text).tolist()[0]
                
                # Execute graph query for exact matches
                graph_results = self.graph_db.execute_query(cypher_query)
                
                # Query vector index for semantic matches
                vector_results = self.graph_db.execute_query("""
                CALL db.index.vector.queryNodes($index_name, $k, $query) 
                YIELD node, score
                RETURN node, score
                """, {
                    "index_name": "memory_embeddings",
                    "k": 5,  # Number of similar results to return
                    "query": query_embedding
                })
                
                # Combine and deduplicate results
                seen_ids = set()
                current_results = []
                
                # Process graph results
                for result in graph_results:
                    node_id = result['n'].id
                    if node_id not in seen_ids:
                        seen_ids.add(node_id)
                        current_results.append(result)
                
                # Add vector results
                for result in vector_results:
                    node_id = result['node'].id
                    if node_id not in seen_ids and result['score'] >= similarity_threshold:
                        seen_ids.add(node_id)
                        current_results.append({
                            'n': result['node'],
                            'score': result['score']
                        })
                
                # Evaluate results
                evaluation = self._evaluate_results(query_text, current_results)
                
                # Update best results if current results are better
                if current_results:
                    best_results = current_results
                
                # Check if results are sufficient
                if evaluation["sufficient"]:
                    self.logger.info(f"Found sufficient results after {iteration + 1} iterations")
                    break
                
                # Update query for next iteration
                if "improved_query" in evaluation:
                    cypher_query = evaluation["improved_query"]
                else:
                    break
                    
                iteration += 1
                
            except Exception as e:
                self.logger.error(f"Error during recall iteration {iteration}: {str(e)}")
                break
        
        return best_results

    def _setup_vector_indexes(self):
        """Set up vector indexes for embedding search"""
        try:
            # Create vector index for embeddings if it doesn't exist
            self.graph_db.execute_query("""
            CREATE VECTOR INDEX memory_embeddings IF NOT EXISTS 
            FOR (n:Memory) ON n.embedding
            OPTIONS {
                indexConfig: {
                    "vector.dimensions": 1536,
                    "vector.similarity_function": "cosine"
                }
            }
            """)
            self.logger.info("Vector index created/verified for embeddings")
        except Exception as e:
            self.logger.error(f"Failed to create vector index: {str(e)}")
            raise

    def _setup_indexes(self):
        """Set up text indexes for searchable fields"""
        prompt = """
        Based on the schema below, determine which fields should be indexed for text search.
        Return a JSON object in a code block with index configurations per node label.
        Only include fields that would be useful for semantic search.

        Schema:
        {self.memory_instructions}

        Return format:
        ```json
        {
            "indexes": [
                {
                    "label": "NodeLabel",
                    "properties": ["field1", "field2"]
                }
            ]
        }
        ```
        """
        
        try:
            response = self.llm_service.get_response([{
                "role": "user",
                "content": prompt.format(self=self)
            }])
            
            snippets = self.code_extractor.extract_snippets(response)
            if snippets and snippets[0].content:
                index_config = eval(snippets[0].content)
                
                for idx in index_config["indexes"]:
                    self.graph_db.create_text_index(
                        idx["label"],
                        idx["properties"]
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to setup indexes: {str(e)}")
