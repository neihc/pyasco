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
    
    # Example conversation to store
    conversation = """
    User: How do I create a virtual environment in Python?
    Assistant: To create a virtual environment in Python, you can use the following steps:
    1. Open your terminal
    2. Navigate to your project directory
    3. Run 'python -m venv myenv'
    4. Activate it using 'source myenv/bin/activate' on Unix/macOS or 'myenv\\Scripts\\activate' on Windows
    
    User: What's the difference between pip and conda?
    Assistant: Pip and conda have several key differences:
    - Pip is Python's package installer, while conda is a package manager for any software
    - Conda can manage different Python versions, pip cannot
    - Conda handles dependencies at the environment level, pip handles them at the package level
    - Conda packages are binaries, while pip typically builds packages from source
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
        "How do I set up Python environments?",
        "What are package managers in Python?",
        "How do I install packages?"
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
