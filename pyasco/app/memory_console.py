"""
PyAsco Memory Console - A tool for managing AI memory

This module provides a console interface for managing PyAsco's memory system,
allowing users to store memories, retrieve context, and trigger memory decay.

Usage:
    python -m pyasco.app.memory_console [options]

Options:
    --config PATH     Path to YAML configuration file
    --db-path PATH   Path to memory database (default: ~/.pyasco/memories)
"""

from typing import Optional
import argparse
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from ..config import ConfigManager
from ..services.lance_memory import LanceDBMemoryHandler
from ..services.embedding import EmbeddingService
from ..services.llm import LLMService
from ..agent.memory_manager import MemoryManager

console = Console()

class MemoryConsole:
    def __init__(self, config_path: Optional[str] = None, db_path: Optional[str] = None):
        # Load configuration
        if config_path:
            config = ConfigManager.load_from_yaml(config_path)
        else:
            # Create minimal config for memory operations
            parser = argparse.ArgumentParser()
            parser.add_argument("--db-path", default="~/.pyasco/memories")
            args = parser.parse_args([])
            if db_path:
                args.db_path = db_path
            config = ConfigManager.from_args(args)

        # Initialize services
        self.embedding_service = EmbeddingService()
        self.memory_handler = LanceDBMemoryHandler(
            db_path=config.memory.db_path if hasattr(config.memory, 'db_path') else db_path or "~/.pyasco/memories"
        )
        self.llm_service = LLMService(
            api_key=config.llm.api_key if hasattr(config, 'llm') else None,
            base_url=config.llm.base_url if hasattr(config, 'llm') else "https://openrouter.ai/api/v1",
            model=config.llm.model if hasattr(config, 'llm') else "meta-llama/llama-3.3-70b-instruct"
        )
        self.memory_manager = MemoryManager(
            memory_handler=self.memory_handler,
            llm_service=self.llm_service
        )

    async def remember(self, content: str) -> None:
        """Store a new memory"""
        try:
            result = await self.memory_manager.remember(content)
            console.print(f"[green]Successfully stored memory:[/green]\n{result}")
        except Exception as e:
            console.print(f"[red]Error storing memory: {str(e)}[/red]")

    async def get_context(self, query: str) -> None:
        """Retrieve context based on query"""
        try:
            context = await self.memory_manager.get_context(query)
            if context:
                console.print("\n[blue]Retrieved context:[/blue]")
                console.print(context)
            else:
                console.print("[yellow]No relevant context found[/yellow]")
        except Exception as e:
            console.print(f"[red]Error retrieving context: {str(e)}[/red]")

    async def trigger_decay(self) -> None:
        """Trigger memory decay process"""
        try:
            await self.memory_manager.trigger_decay()
            console.print("[green]Successfully triggered memory decay process[/green]")
        except Exception as e:
            console.print(f"[red]Error during memory decay: {str(e)}[/red]")

async def main():
    parser = argparse.ArgumentParser(description="PyAsco Memory Management Console")
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--db-path", help="Path to memory database")
    args = parser.parse_args()

    memory_console = MemoryConsole(args.config, args.db_path)

    console.print("[bold blue]Welcome to PyAsco Memory Console![/bold blue]")
    console.print("Commands:")
    console.print("  1. remember - Store a new memory")
    console.print("  2. context - Get context based on query")
    console.print("  3. decay - Trigger memory decay process")
    console.print("  4. exit - Quit the console\n")

    while True:
        command = Prompt.ask("Command").lower()

        if command == "exit" or command == "4":
            break
        elif command == "remember" or command == "1":
            content = Prompt.ask("Enter memory content")
            await memory_console.remember(content)
        elif command == "context" or command == "2":
            query = Prompt.ask("Enter search query")
            await memory_console.get_context(query)
        elif command == "decay" or command == "3":
            await memory_console.trigger_decay()
        else:
            console.print("[yellow]Invalid command[/yellow]")

if __name__ == "__main__":
    asyncio.run(main())
