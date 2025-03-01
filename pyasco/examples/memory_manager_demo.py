import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

from pyasco.services.llm import LLMService
from ..services.lance_memory import LanceDBMemoryHandler, MemoryType
from ..agent.memory_manager import MemoryManager
from ..config import Config, ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_memory_operations(memory_manager: MemoryManager):
    """Demonstrate memory operations including decay process"""
    
    web_context = await memory_manager.get_context(
            "get my rabbitmq stats"
    )
    logger.info(f"Web development context:\n{web_context}")

async def main():
    """Main demo function"""
    # Create temporary demo database path
    # Initialize memory handler with demo database
    memory_handler = LanceDBMemoryHandler()

    # Create memory manager
    llm_service = LLMService()
    memory_manager = MemoryManager(memory_handler, llm_service)

    # Run demo operations
    await demonstrate_memory_operations(memory_manager)
        

if __name__ == "__main__":
    asyncio.run(main())
