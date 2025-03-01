"""Predefined functions and utilities for the code executor environment"""

import os
import sys
import json
import base64
import asyncio
from typing import Any, Dict, List
from ...services.lance_memory import LanceDBMemoryHandler
from ...services.embedding import EmbeddingService
from ...services.llm import LLMService
from ...agent.memory_manager import MemoryManager

def load_json( str) -> Dict:
    """Load JSON data safely"""
    return json.loads(data)

def save_json( Any) -> str:
    """Save data as JSON string"""
    return json.dumps(data, indent=2)

def encode_base64( str) -> str:
    """Encode string as base64"""
    return base64.b64encode(data.encode()).decode()

def decode_base64( str) -> str:
    """Decode base64 string"""
    return base64.b64decode(data.encode()).decode()

async def search_context(query: str) -> List[Dict]:
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
    context = await memory_manager.get_context(query, raw=True)
    
    # Sort by score in descending order
    return sorted(context, key=lambda x: x.get('final_score', 0), reverse=True)

# Add more predefined functions as needed
