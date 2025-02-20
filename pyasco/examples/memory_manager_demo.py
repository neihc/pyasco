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
    """Demonstrate memory operations including decay process"""
    
    # First conversation about programming languages
    logger.info("Storing memories about programming languages...")
    prog_lang_convo = [
        "User discussed Python's readability principles",
        "Agent explained Python's 'batteries included' philosophy",
        "User compared Python to JavaScript ecosystem",
        "Agent discussed Python's strong typing options",
        "User mentioned experience with type hints in large projects"
    ]
    
    # Store initial memories
    for memory in prog_lang_convo:
        memory_id = await memory_manager.remember(
            content=memory,
            meta={
                "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
                "topic": "programming_languages"
            }
        )
        logger.info(f"Stored old memory: {memory}")
    
    # Second conversation about web development
    logger.info("\nStoring memories about web development...")
    web_dev_convo = [
        "User asked about modern frontend frameworks",
        "Agent compared React, Vue, and Svelte",
        "User shared experience with React hooks",
        "Agent discussed state management patterns",
        "User mentioned interest in server components"
    ]
    
    # Store more recent memories
    for memory in web_dev_convo:
        memory_id = await memory_manager.remember(
            content=memory,
            meta={
                "timestamp": datetime.now().isoformat(),
                "topic": "web_development"
            }
        )
        logger.info(f"Stored recent memory: {memory}")
    
    # Query before decay
    logger.info("\nQuerying about programming languages before decay:")
    before_context = await memory_manager.get_context(
        "What does the user know about programming languages?"
    )
    logger.info(f"Context before decay:\n{before_context}")
    
    # Trigger memory decay process
    logger.info("\nTriggering memory decay process...")
    await memory_manager.trigger_decay()
    
    # Query after decay
    logger.info("\nQuerying about programming languages after decay:")
    after_context = await memory_manager.get_context(
        "What does the user know about programming languages?"
    )
    logger.info(f"Context after decay:\n{after_context}")
    
    # Show how recent memories are still intact
    logger.info("\nQuerying about web development (should be unchanged):")
    web_context = await memory_manager.get_context(
        "What does the user know about web development?"
    )
    logger.info(f"Web development context:\n{web_context}")

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
