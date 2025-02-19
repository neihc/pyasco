import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType
from ..agent.memory_manager import MemoryManager
from ..config import Config, ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_memory_operations(memory_manager: MemoryManager):
    """Demonstrate basic memory operations"""
    
    # Store some short-term memories
    logger.info("Storing short-term memories...")
    memories = [
        "The user is working on a Python project using asyncio",
        "The project involves natural language processing",
        "The user prefers using type hints in their code",
        "The last conversation was about memory management",
    ]
    
    for memory in memories:
        memory_id = await memory_manager.remember(
            content=memory,
            meta={"timestamp": datetime.now().isoformat()}
        )
        logger.info(f"Stored memory with ID: {memory_id}")
    
    # Get context for a query
    query = "What programming preferences does the user have?"
    logger.info(f"\nGetting context for query: {query}")
    context = await memory_manager.get_context(query)
    logger.info(f"Retrieved context:\n{context}")
    
    # Add some older memories to demonstrate decay
    logger.info("\nAdding older memories...")
    old_memories = [
        "The user previously worked on a Django project",
        "The user mentioned they like VS Code",
    ]
    
    for memory in old_memories:
        # Set creation time to 8 days ago to trigger decay
        old_timestamp = (datetime.now() - timedelta(days=8)).isoformat()
        memory_id = await memory_manager.remember(
            content=memory,
            meta={"timestamp": old_timestamp}
        )
        logger.info(f"Stored old memory with ID: {memory_id}")
    
    # Force decay check by adding a new memory
    logger.info("\nTriggering memory decay...")
    await memory_manager.remember(
        "This new memory will trigger decay of old memories",
        meta={"timestamp": datetime.now().isoformat()}
    )
    
    # Get updated context
    logger.info("\nGetting updated context after decay:")
    context = await memory_manager.get_context(query)
    logger.info(f"Retrieved context:\n{context}")

async def main():
    """Main demo function"""
    # Initialize with default config
    config = Config(
        llm_config=None,  # Not needed for this demo
        memory_config=None,  # Will use defaults
        embedding_config=None,  # Will use defaults
        docker_config=None,  # Not needed for this demo
        graphdb_config=None,  # Not needed for this demo
    )
    
    # Initialize memory handler
    memory_handler = LanceDBMemoryHandler()
    await memory_handler._initialize_db()
    
    # Create memory manager
    memory_manager = MemoryManager(memory_handler)
    
    try:
        await demonstrate_memory_operations(memory_manager)
    except Exception as e:
        logger.error(f"Error during demo: {e}")
        raise
    finally:
        # Cleanup
        await memory_handler.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
