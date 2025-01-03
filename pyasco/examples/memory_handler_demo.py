from pyasco.services.graphdb import GraphDB
from pyasco.services.llm import LLMService
from pyasco.agent.memory_handler import MemoryHandler

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
        - category: type of memory
    - Entity
        - name: name of person, place, or thing
        - type: what kind of entity (person/place/thing)
    
    Relationships:
    - MENTIONS: connects Memory to Entity
    - RELATES_TO: connects Memory to other Memory nodes
    """
    
    # Initialize memory handler
    memory_handler = MemoryHandler(
        memory_instructions=memory_instructions,
        llm_service=llm_service,
        graph_db=graph_db
    )
    
    # Example memory to store
    content = """
    Yesterday I had a great meeting with Alice about the new project. 
    We discussed implementation details at the coffee shop downtown.
    She suggested using Python for the backend development.
    """
    
    context = {
        "timestamp": "2024-01-03T14:30:00",
        "source": "user_input"
    }
    
    try:
        # Store the memory
        stored_memory = memory_handler.remember(content, context)
        print("Successfully stored memory:")
        print(f"Node properties: {stored_memory}")
        
    except Exception as e:
        print(f"Error storing memory: {e}")
    
    finally:
        # Clean up connections
        graph_db.close()

main()
