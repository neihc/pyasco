from typing import Dict, Any, List, Optional
from ..services.graphdb import GraphDB
from ..services.llm import LLMService
from ..services.code_snippet_extractor import CodeSnippetExtractor
from ..logger_config import setup_logger

class MemoryHandler:
    """Handler for processing and storing memories using LLM and graph database"""
    
    def __init__(self, memory_instructions: str, llm_service: Optional[LLMService] = None,
                 graph_db: Optional[GraphDB] = None):
        """
        Initialize the memory handler
        Args:
            memory_instructions (str): Instructions for how to process and structure memories
            llm_service (LLMService): LLM service instance for processing memories
            graph_db (GraphDB): GraphDB instance for storing memories
        """
        self.logger = setup_logger('memory_handler')
        self.memory_instructions = memory_instructions
        self.llm_service = llm_service or LLMService()
        self.graph_db = graph_db or GraphDB()
        self.code_extractor = CodeSnippetExtractor()

    def remember(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process content and store it as a memory in the graph database
        Args:
            content (str): The content to be remembered
            context (dict): Optional contextual information
        Returns:
            dict: The stored memory node properties
        """
        # Prepare prompt for LLM to structure the memory
        prompt = f"""
        Process the following content into a structured memory format.
        
        CONTENT:
        {content}
        
        CONTEXT:
        {context or {}}
        
        INSTRUCTIONS:
        {self.memory_instructions}
        
        Based on the provided schema in the instructions, structure this content as a JSON object.
        You should:
        1. Determine appropriate node types and their properties
        2. Identify any relationships that should be created
        3. Return a JSON object wrapped in a code block like this:
        ```json
        {
            "nodes": [
                {
                    "label": "node type from schema",
                    "properties": {
                        "property1": "value1",
                        ...
                    }
                }
            ],
            "relationships": [
                {
                    "from_node_id": "unique_id_1",
                    "to_node_id": "unique_id_2", 
                    "type": "RELATIONSHIP_TYPE",
                    "properties": {
                        "property1": "value1",
                        ...
                    }
                }
            ]
        }
        ```
        """

        # Get structured memory from LLM
        try:
            response = self.llm_service.get_response([{
                "role": "user",
                "content": prompt
            }])
            
            # Extract JSON from code block
            snippets = self.code_extractor.extract_snippets(response)
            if not snippets or not snippets[0].content:
                raise ValueError("No JSON structure found in LLM response")
                
            memory_structure = eval(snippets[0].content)
            
            # Create all nodes first
            created_nodes = []
            for node_spec in memory_structure["nodes"]:
                # Add original content to first node's properties
                if not created_nodes:
                    node_spec["properties"]["original_content"] = content
                    if context:
                        node_spec["properties"].update(context)
                
                node = self.graph_db.create_node(
                    node_spec["label"],
                    node_spec["properties"]
                )
                created_nodes.append(node)
            
            # Create relationships between nodes
            for rel in memory_structure["relationships"]:
                self.graph_db.create_relationship(
                    rel["from_node_id"],
                    rel["to_node_id"],
                    rel["type"],
                    rel.get("properties", {})
                )

            self.logger.info(f"Successfully stored memory with {len(created_nodes)} nodes")
            return created_nodes[0]  # Return the primary node

        except Exception as e:
            self.logger.error(f"Failed to process memory: {str(e)}")
            raise
