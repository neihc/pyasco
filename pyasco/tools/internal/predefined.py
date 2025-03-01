"""Predefined functions and utilities for the code executor environment"""

from pyasco.services.lance_memory import LanceDBMemoryHandler
from pyasco.services.embedding import EmbeddingService
from pyasco.services.llm import LLMService
from pyasco.agent.memory_manager import MemoryManager

async def search_context(query: str) -> str:
    """Search memory context using the given query
    
    Args:
        query: Search query string
        
    Returns:
        List of memory entries with scores and content
    """
    # Initialize required services
    memory_handler = LanceDBMemoryHandler()
    embedding_service = EmbeddingService()
    llm_service = LLMService()
    
    # Create memory manager
    memory_manager = MemoryManager(
        memory_handler=memory_handler,
        llm_service=llm_service
    )
    
    # Get context with raw=True to get full memory entries
    context = await memory_manager.get_context(query, raw=False)
    return context
