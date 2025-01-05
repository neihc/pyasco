from pyasco.services.graphdb import GraphDB
from pyasco.services.llm import LLMService
from pyasco.agent.memory_handler import MemoryHandler
from datetime import datetime, timedelta
import json

def main():
    # Initialize services
    graph_db = GraphDB(
        uri="neo4j+s://your-neo4j-instance.io",
        username="neo4j",
        password="your-password"
    )
    
    llm_service = LLMService()
    
    # Initialize memory handler with custom schema
    memory_instructions = """
    Additional schema for AI agent memories:
    
    Nodes:
    - Topic
        - name: subject of discussion
        - domain: technical/personal/project
        - complexity: basic/intermediate/advanced
        
    - Skill
        - name: specific capability or knowledge
        - proficiency: beginner/intermediate/expert
        - last_used: timestamp
        
    Relationships:
    - REQUIRES_SKILL: connects Topic to required Skills
    - BUILDS_ON: shows prerequisite relationships
    - RELATED_TO: connects similar or related Topics
    """
    
    memory_handler = MemoryHandler(
        memory_instructions=memory_instructions,
        llm_service=llm_service,
        graph_db=graph_db
    )
    
    try:
        # Store a conversation memory
        conversation = memory_handler.remember(
            """The user asked about implementing BERT for text classification. 
            We discussed fine-tuning approaches and potential pitfalls. 
            They showed good understanding of transformer architecture but needed 
            guidance on handling long sequences.""",
            {
                "timestamp": datetime.now(),
                "user_id": "user123",
                "conversation_id": "conv456"
            }
        )
        print("\nStored conversation memory:")
        print(json.dumps(conversation, indent=2))
        
        # Demonstrate memory recall
        print("\nRecalling memories about transformers:")
        transformer_memories = memory_handler.recall(
            "What discussions have we had about transformer models?",
            node_types=["Conversation", "Message"]
        )
        print(json.dumps(transformer_memories, indent=2))
        
        print("\nRecalling user preferences:")
        preference_memories = memory_handler.recall(
            "What are the user's learning preferences and skill levels?",
            node_types=["User", "Skill"]
        )
        print(json.dumps(preference_memories, indent=2))
        
    except Exception as e:
        print(f"Error in memory operations: {e}")
        
    finally:
        graph_db.close()

if __name__ == "__main__":
    main()
