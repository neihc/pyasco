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
            
            # Create memory node in graph database
            memory_node = graphdb.create_node("Memory", {
                "content": content,
                "structured_data": memory_structure,
                **context if context else {}
            })
            
            # Process relationships if any
            if memory_node and "relationships" in memory_structure:
                for rel in memory_structure["relationships"]:
                    # Create or find target node
                    target_node = graphdb.create_node(
                        rel["type"].capitalize(),
                        {"name": rel["target"]}
                    )
                    
                    # Create relationship
                    if target_node:
                        graphdb.create_relationship(
                            memory_node["id"],
                            target_node["id"],
                            rel["type"].upper(),
                            rel.get("properties", {})
                        )

            self.logger.info(f"Successfully stored memory: {memory_structure['summary']}")
            return memory_node

        except Exception as e:
            self.logger.error(f"Failed to process memory: {str(e)}")
            raise
