import os
import argparse
from typing import Optional
import yaml

from ..config import Config, ConfigManager
from ..services.llm import LLMService
from ..services.embedding import EmbeddingService
from ..services.lance_memory import LanceDBMemoryHandler

def demonstrate_memory_operations(memory_handler: LanceDBMemoryHandler, debug: bool = False):
    """Demonstrate memory storage and retrieval operations"""
    
    print("\n=== Storing New Memories ===")
    # Store initial memory about user preferences
    initial_memory = """
    User: I prefer working late at night because it's quieter and I can focus better.
    Assistant: I understand. Your preference for night work seems related to:
    - Reduced distractions during quiet hours
    - Better focus and concentration
    - More peaceful work environment
    """
    result = memory_handler.remember(initial_memory, context={"source": "chat"})
    print("Stored initial memory about work preferences")
    if debug:
        print(f"Result: {result}")

    # Store memory about learning style
    learning_memory = """
    User: I find that I learn programming concepts better when I build small projects.
    Assistant: That's a valuable insight about your learning style:
    - Hands-on learning through practical projects
    - Active engagement rather than passive reading
    - Real-world application of concepts
    """
    result = memory_handler.remember(learning_memory, context={"source": "chat"})
    print("Stored memory about learning preferences")

    print("\n=== Updating Existing Memory ===")
    # Update with more specific information
    update_memory = """
    User: Actually, I specifically prefer coding between 10 PM and 2 AM, and I'm most productive 
    during these hours. The complete silence helps me solve complex problems.
    """
    # First recall existing memory to update
    existing = memory_handler.recall("working late at night", limit=1)
    if existing:
        result = memory_handler.remember(update_memory, 
                                       context={"source": "chat", "update": True},
                                       related_memories=existing)
        print("Updated work schedule preferences with specific hours")

    print("\n=== Adding New Related Memory ===")
    # Add related information
    additional_memory = """
    User: I've also noticed I need at least 7 hours of sleep to maintain this schedule,
    so I usually wake up around 9 AM to stay productive.
    """
    result = memory_handler.remember(additional_memory, 
                                   context={"source": "chat"},
                                   related_memories=existing)
    print("Added sleep schedule information")

    # Example queries to demonstrate recall
    print("\n=== Recalling Memories ===")
    queries = [
        "What are the user's work preferences?",
        "How does the user learn best?",
        "What do we know about the user's sleep schedule?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        memories = memory_handler.recall(query, limit=2)
        for memory in memories:
            print(f"\nRelevant Memory (similarity: {memory['similarity']:.2f}):")
            print(memory['content'])
            if debug:
                print(f"Created at: {memory['created_at']}")
                print(f"Meta {memory['metadata']}")

def main():
    parser = argparse.ArgumentParser(description='Demonstrate DuckDB Memory Handler')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    args = parser.parse_args()

    # Load configuration
    config = ConfigManager.load_from_yaml(args.config)
    
    # Initialize services
    llm_service = LLMService()
    embedding_service = EmbeddingService()
    
    # Initialize memory handler with custom path for demo
    memory_handler = LanceDBMemoryHandler(
        llm_service=llm_service,
        embedding_service=embedding_service,
        db_path="demo_memories.lance"
    )
    
    try:
        demonstrate_memory_operations(memory_handler, debug=args.debug)
    finally:
        # Cleanup
        if os.path.exists("demo_memories.lance"):
            import shutil
            shutil.rmtree("demo_memories.lance")

if __name__ == "__main__":
    main()
