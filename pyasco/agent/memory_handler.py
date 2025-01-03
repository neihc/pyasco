from typing import Dict, Any, List, Optional
from ..services import graphdb, llm
from ..logger_config import setup_logger

class MemoryHandler:
    """Handler for processing and storing memories using LLM and graph database"""
    
    def __init__(self, memory_instructions: str):
        """
        Initialize the memory handler
        Args:
            memory_instructions (str): Instructions for how to process and structure memories
        """
        self.logger = setup_logger('memory_handler')
        self.memory_instructions = memory_instructions

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
        
        Format your response as a JSON object with these fields:
        {{
            "type": "memory type (conversation, fact, skill, etc)",
            "summary": "brief summary of the memory",
            "details": "detailed content of the memory",
            "tags": ["relevant", "tags", "for", "categorization"],
            "relationships": [
                {{
                    "type": "relationship type",
                    "target": "related memory or concept",
                    "properties": {{}}
                }}
            ]
        }}
        """

        # Get structured memory from LLM
        try:
            memory_structure = llm.get_openai_response([{
                "role": "user",
                "content": prompt
            }])
            
            # Create all nodes first
            created_nodes = []
            for node_spec in memory_structure["nodes"]:
                # Add original content to first node's properties
                if not created_nodes:
                    node_spec["properties"]["original_content"] = content
                    if context:
                        node_spec["properties"].update(context)
                
                node = graphdb.create_node(
                    node_spec["label"],
                    node_spec["properties"]
                )
                created_nodes.append(node)
            
            # Create relationships between nodes
            for rel in memory_structure["relationships"]:
                from_node = created_nodes[rel["from_node_index"]]
                to_node = created_nodes[rel["to_node_index"]]
                
                graphdb.create_relationship(
                    from_node["id"],
                    to_node["id"],
                    rel["type"],
                    rel.get("properties", {})
                )

            self.logger.info(f"Successfully stored memory with {len(created_nodes)} nodes")
            return created_nodes[0]  # Return the primary node

        except Exception as e:
            self.logger.error(f"Failed to process memory: {str(e)}")
            raise
