from typing import Dict, Any, List, Optional
from ..services.graphdb import GraphDB
from ..services.llm import LLMService
from ..services.code_snippet_extractor import CodeSnippetExtractor
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
                 graph_db: Optional[GraphDB] = None):
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
        self._setup_indexes()

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

    def recall(self, query: str, node_types: Optional[List[str]] = None,
               limit: int = 5) -> List[Dict[str, Any]]:
        """
        Recall memories using entity extraction and iterative expansion
        Args:
            query (str): Natural language query to search memories
            node_types (list): Optional list of node types to search within
            limit (int): Maximum number of results to return
        Returns:
            list: List of relevant memory nodes with similarity scores
        """
        try:
            # Phase 1: Extract entities
            entities = self._extract_entities(query)
            self.logger.debug(f"Extracted entities: {entities}")
            
            # Phase 2: Iteratively expand search until results are found
            results = []
            for entity in entities:
                while not results:
                    results = self._search_entity(entity, node_types, limit)
                    if not results:
                        self.logger.debug(f"No results for entity '{entity}', expanding search...")
                        entity = self._expand_entity(entity)
            
            self.logger.info(f"Found {len(results)} memories after processing")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to recall memories: {str(e)}")
            raise

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
