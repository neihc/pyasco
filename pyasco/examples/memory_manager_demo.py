import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType
from ..agent.memory_manager import MemoryManager
from ..config import Config, ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_memory_operations(memory_manager: MemoryManager):
    """Demonstrate basic memory operations with a simulated conversation"""
    
    # Simulate a conversation about web frameworks
    logger.info("Simulating conversation about web frameworks...")
    web_framework_convo = [
        "User asked about differences between Django and FastAPI",
        "Agent explained that Django is a full-featured framework while FastAPI is more lightweight",
        "User mentioned they prefer FastAPI's async capabilities",
        "Agent discussed FastAPI's modern features like automatic OpenAPI docs",
        "User shared their experience with FastAPI's dependency injection system"
    ]
    
    for memory in web_framework_convo:
        memory_id = await memory_manager.remember(
            content=memory,
            meta={"timestamp": datetime.now().isoformat(), "topic": "web_frameworks"}
        )
        logger.info(f"Stored conversation memory: {memory}")
    
    # Query about web framework preferences
    query = "What does the user think about web frameworks?"
    logger.info(f"\nGetting context for query: {query}")
    context = await memory_manager.get_context(query)
    logger.info(f"Retrieved context:\n{context}")
    
    # Simulate time passing (8 days) and new conversation about a different topic
    logger.info("\nSimulating new conversation after time passage...")
    
    # New conversation about data science
    data_science_convo = [
        "User is starting a new data science project",
        "They're considering using pandas and scikit-learn",
        "Agent suggested using jupyter notebooks for exploration",
        "User mentioned they prefer VS Code's jupyter integration"
    ]
    
    # Add new memories with current timestamp
    for memory in data_science_convo:
        memory_id = await memory_manager.remember(
            content=memory,
            meta={"timestamp": datetime.now().isoformat(), "topic": "data_science"}
        )
        logger.info(f"Stored new memory: {memory}")
    
    # Query about both topics to see decay effects
    logger.info("\nQuerying about web frameworks (should show decay):")
    web_context = await memory_manager.get_context("What does the user know about web frameworks?")
    logger.info(f"Web frameworks context:\n{web_context}")
    
    logger.info("\nQuerying about data science (should be fresh):")
    ds_context = await memory_manager.get_context("What is the user's data science experience?")
    logger.info(f"Data science context:\n{ds_context}")

async def main():
    """Main demo function"""
    # Create temporary demo database path
    demo_db_path = Path.home() / ".pyasco" / "demo_memories"
    
    try:
        # Initialize memory handler with demo database
        memory_handler = LanceDBMemoryHandler(db_path=str(demo_db_path))
        
        # Create memory manager
        memory_manager = MemoryManager(memory_handler)
        
        # Run demo operations
        await demonstrate_memory_operations(memory_manager)
        
    except Exception as e:
        logger.error(f"Error during demo: {e}")
        raise
    finally:
        # Clean up demo database
        if demo_db_path.exists():
            shutil.rmtree(demo_db_path)
            logger.info(f"Cleaned up demo database at {demo_db_path}")

if __name__ == "__main__":
    asyncio.run(main())
