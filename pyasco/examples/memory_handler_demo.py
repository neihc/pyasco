from pyasco.services.graphdb import GraphDB
from pyasco.services.llm import LLMService
from pyasco.agent.memory_handler import MemoryHandler
from datetime import datetime, timedelta

def main():
    # Initialize services
    graph_db = GraphDB(
        uri="neo4j+s://61e1f96d.databases.neo4j.io",
        username="neo4j",
        password="pEBDsNRqWZvwTI2IooyrQRuNX__vYmmgqcef9MPOY0g"  # Replace with actual password
    )
    
    llm_service = LLMService()  # Using default configuration
    
    # Define memory processing instructions
    memory_instructions = """
    Schema for processing memories:
    
    Nodes:
    - Memory (main node)
        - content: text of the memory
        - timestamp: when it was created
        - category: type of memory (conversation, task, learning, preference)
        - importance: high/medium/low
    - User
        - name: user's name
        - role: their role in the interaction
    - Topic
        - name: subject matter
        - domain: general category
    
    Relationships:
    - INTERACTS_WITH: connects Memory to User
    - DISCUSSES: connects Memory to Topic
    - FOLLOWS: temporal relationship between Memory nodes
    - REFERENCES: when one memory refers to another
    """
    
    # Initialize memory handler
    memory_handler = MemoryHandler(
        memory_instructions=memory_instructions,
        llm_service=llm_service,
        graph_db=graph_db
    )
    
    # Series of interactions to remember
    memories = [
        {
            "content": """
            User John introduced himself and mentioned he's working on a machine learning project.
            He expressed interest in natural language processing and asked about transformer architectures.
            I provided a detailed explanation of attention mechanisms.
            """,
            "context": {
                "timestamp": datetime.now() - timedelta(days=2),
                "category": "conversation",
                "user": "John"
            }
        },
        {
            "content": """
            John returned to discuss his progress. He implemented the transformer model we discussed
            but encountered issues with training on his dataset. I suggested using gradient checkpointing
            and reducing batch size to handle memory constraints.
            """,
            "context": {
                "timestamp": datetime.now() - timedelta(days=1),
                "category": "technical_support",
                "user": "John",
                "references_previous": True
            }
        },
        {
            "content": """
            Today's session with John was successful. He reported that the memory optimizations worked,
            and his model is now training properly. He showed particular interest in attention visualization
            techniques, which we explored in detail. He prefers practical examples over theoretical explanations.
            """,
            "context": {
                "timestamp": datetime.now(),
                "category": "learning",
                "user": "John",
                "user_preferences": ["practical_examples", "visual_explanations"]
            }
        }
    ]
    
    try:
        # Store multiple memories
        for memory in memories:
            stored_memory = memory_handler.remember(memory["content"], memory["context"])
            print(f"\nSuccessfully stored memory from {memory['context']['timestamp']}:")
            print(f"Node properties: {stored_memory}")
            
    except Exception as e:
        print(f"Error storing memories: {e}")
    
    finally:
        # Clean up connections
        graph_db.close()

if __name__ == "__main__":
    main()
