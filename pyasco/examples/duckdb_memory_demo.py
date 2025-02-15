import os
import argparse
from typing import Optional
import yaml

from ..config import Config, ConfigManager
from ..services.llm import LLMService
from ..services.embedding import EmbeddingService
from ..services.duckdb_memory import DuckDBMemoryHandler

def demonstrate_memory_operations(memory_handler: DuckDBMemoryHandler, debug: bool = False):
    """Demonstrate memory storage and retrieval operations"""
    
    # Example user insights to store
    conversation = """
    User: I prefer working late at night because it's quieter and I can focus better.
    Assistant: I understand. Your preference for night work seems related to:
    - Reduced distractions during quiet hours
    - Better focus and concentration
    - More peaceful work environment
    
    User: I find that I learn programming concepts better when I build small projects.
    Assistant: That's a valuable insight about your learning style:
    - Hands-on learning through practical projects
    - Active engagement rather than passive reading
    - Real-world application of concepts
    - Learning through trial and error
    """
    
    print("\n=== Storing Memories ===")
    result = memory_handler.remember(conversation, {"source": "example_chat"})
    print(f"Stored {len(result['memories'])} memories")
    if debug:
        for memory in result['memories']:
            print(f"\nMemory: {memory['content']}")
            print(f"Meta {memory['metadata']}")

    # Example queries to demonstrate recall
    queries = [
        "How does the user prefer to work?",
        "What's the user's learning style?",
        "When is the user most productive?"
    ]
    
    print("\n=== Recalling Memories ===")
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
    memory_handler = DuckDBMemoryHandler(
        llm_service=llm_service,
        embedding_service=embedding_service,
        db_path="demo_memories.db"
    )
    
    try:
        demonstrate_memory_operations(memory_handler, debug=args.debug)
    finally:
        # Cleanup
        if os.path.exists("demo_memories.db"):
            os.remove("demo_memories.db")

if __name__ == "__main__":
    main()
