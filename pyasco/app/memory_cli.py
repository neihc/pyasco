#!/usr/bin/env python3
"""
Memory CLI - A command-line interface for managing LanceDB memories
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.progress import Progress

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pyasco.services.lance_memory import LanceDBMemoryHandler, MemoryType


class QueryType(str, Enum):
    """Types of search queries"""
    VECTOR = "vector"
    FTS = "fts"
    HYBRID = "hybrid"


class MemoryCLI:
    """CLI for managing LanceDB memories"""
    
    def __init__(self, db_path: str = "~/.pyasco/memories"):
        self.console = Console()
        self.memory_handler = LanceDBMemoryHandler(db_path=db_path)
        # Available commands
        self.commands = [
            'search', 'add', 'delete', 'update', 'tags', 'stats', 'help', 'exit'
        ]
        
        # Memory types
        self.memory_types = [t.value for t in MemoryType]
        
        # Query types 
        self.query_types = [t.value for t in QueryType]
        
    async def start(self):
        """Start the CLI interface"""
        self.console.print(Panel.fit(
            "[bold cyan]Memory CLI[/bold cyan]\n"
            "[dim]A command-line interface for managing LanceDB memories[/dim]",
            border_style="cyan"
        ))
        
        self.console.print("\nType [bold]help[/bold] to see available commands.")
        
        while True:
            try:
                command = input("memory> ").strip()
                
                if not command:
                    continue
                    
                if command == "exit":
                    break
                    
                await self.process_command(command)
                
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
                
        self.console.print("[cyan]Goodbye![/cyan]")
        
    async def process_command(self, command_line: str):
        """Process a command entered by the user"""
        parts = command_line.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "help":
            self.show_help()
        elif command == "search":
            await self.search_memories(args)
        elif command == "add":
            await self.add_memory()
        elif command == "delete":
            await self.delete_memory(args)
        elif command == "update":
            await self.update_memory(args)
        elif command == "tags":
            await self.show_tags()
        elif command == "stats":
            await self.show_stats()
        else:
            self.console.print(f"[bold red]Unknown command:[/bold red] {command}")
            self.console.print("Type [bold]help[/bold] to see available commands.")
            
    def show_help(self):
        """Show help information"""
        table = Table(title="Available Commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        table.add_column("Usage", style="dim")
        
        table.add_row(
            "search", 
            "Search for memories", 
            "search <query> [--type=hybrid|vector|fts] [--limit=10]"
        )
        table.add_row(
            "add", 
            "Add a new memory", 
            "add"
        )
        table.add_row(
            "delete", 
            "Delete a memory by ID", 
            "delete <memory_id>"
        )
        table.add_row(
            "update", 
            "Update a memory by ID", 
            "update <memory_id>"
        )
        table.add_row(
            "tags", 
            "Show all tags", 
            "tags"
        )
        table.add_row(
            "stats", 
            "Show database statistics", 
            "stats"
        )
        table.add_row(
            "help", 
            "Show this help", 
            "help"
        )
        table.add_row(
            "exit", 
            "Exit the program", 
            "exit"
        )
        
        self.console.print(table)
        
    async def search_memories(self, args_str: str):
        """Search for memories"""
        # Parse arguments
        query = ""
        query_type = "hybrid"
        limit = 10
        
        # Simple argument parsing
        parts = args_str.split()
        if not parts:
            self.console.print("[bold red]Error:[/bold red] Search query required")
            return
            
        # Extract flags
        flags = [p for p in parts if p.startswith("--")]
        query_parts = [p for p in parts if not p.startswith("--")]
        query = " ".join(query_parts)
        
        for flag in flags:
            if flag.startswith("--type="):
                query_type = flag.split("=")[1]
                if query_type not in self.query_types:
                    self.console.print(f"[bold red]Error:[/bold red] Invalid query type: {query_type}")
                    return
            elif flag.startswith("--limit="):
                try:
                    limit = int(flag.split("=")[1])
                except ValueError:
                    self.console.print("[bold red]Error:[/bold red] Limit must be a number")
                    return
        
        # Perform search
        with Progress() as progress:
            task = progress.add_task("[cyan]Searching...", total=1)
            
            results = await self.memory_handler.search_similar(
                query=query,
                limit=limit,
                score_threshold=0.0,
            )
            
            progress.update(task, completed=1)
        
        # Display results
        if not results:
            self.console.print("[yellow]No results found[/yellow]")
            return
            
        # Update access counts for found memories
        memory_ids = [r["id"] for r in results]
        await self.memory_handler.increment_access_count(memory_ids)
        
        # Create table
        table = Table(title=f"Search Results for '{query}' (type: {query_type})")
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Content", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Tags", style="yellow")
        table.add_column("Score", justify="right")
        table.add_column("Created", style="dim")
        
        # Add rows
        for result in results:
            # Format content (truncate if too long)
            content = result["content"]
            if len(content) > 100:
                content = content[:97] + "..."
                
            # Format tags
            tags = ", ".join(result.get("tags", []))
            
            # Format score
            score = result.get("_relevance_score", 0)
            score_str = f"{score:.4f}" if score else "N/A"
            
            # Format date
            created_at = result.get("created_at")
            date_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else "N/A"
            
            table.add_row(
                result["id"][:8],  # Truncate ID for display
                content,
                result.get("memory_type", "unknown"),
                tags,
                score_str,
                date_str
            )
            
        self.console.print(table)
        
        # Ask if user wants to see details of any result
        while True:
            detail_id = input("Enter ID to see details (or press Enter to continue): ").strip()
            
            if not detail_id:
                break
                
            # Find the memory with matching ID prefix
            memory = next((r for r in results if r["id"].startswith(detail_id)), None)
            if memory:
                self.display_memory_details(memory)
            else:
                self.console.print("[bold red]Memory not found[/bold red]")
    
    def display_memory_details(self, memory: Dict[str, Any]):
        """Display detailed information about a memory"""
        # Create a panel with memory details
        content = Text()
        content.append(f"ID: ", style="bold")
        content.append(f"{memory['id']}\n\n")
        
        content.append(f"Type: ", style="bold green")
        content.append(f"{memory.get('memory_type', 'unknown')}\n\n")
        
        content.append(f"Content:\n", style="bold cyan")
        content.append(f"{memory['content']}\n\n")
        
        content.append(f"Tags: ", style="bold yellow")
        content.append(f"{', '.join(memory.get('tags', []))}\n\n")
        
        content.append(f"Created: ", style="bold")
        created_at = memory.get("created_at")
        content.append(f"{created_at.strftime('%Y-%m-%d %H:%M:%S') if created_at else 'N/A'}\n\n")
        
        content.append(f"Access Count: ", style="bold")
        content.append(f"{memory.get('access_count', 0)}\n\n")
        
        content.append(f"Importance Score: ", style="bold")
        content.append(f"{memory.get('importance_score', 0)}\n\n")
        
        # Format metadata as JSON
        content.append(f"Meta\n", style="bold")
        metadata = memory.get("metadata", {})
        if metadata:
            json_str = json.dumps(metadata, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            self.console.print(Panel(content, title=f"Memory Details", border_style="cyan"))
            self.console.print(syntax)
        else:
            content.append("No metadata")
            self.console.print(Panel(content, title=f"Memory Details", border_style="cyan"))
    
    async def add_memory(self):
        """Add a new memory interactively"""
        self.console.print("[bold cyan]Add New Memory[/bold cyan]")
        
        # Get memory content
        print("Enter content (press Ctrl+D or Ctrl+Z on Windows when done):")
        content_lines = []
        try:
            while True:
                line = input()
                content_lines.append(line)
        except EOFError:
            content = "\n".join(content_lines).strip()
        
        if not content:
            self.console.print("[bold red]Error:[/bold red] Content cannot be empty")
            return
            
        # Get memory type
        print(f"Available memory types: {', '.join(self.memory_types)}")
        memory_type = input("Memory Type: ").strip()
        
        if not memory_type or memory_type not in self.memory_types:
            self.console.print(f"[bold red]Error:[/bold red] Invalid memory type. Choose from: {', '.join(self.memory_types)}")
            return
            
        # Get tags
        tags_str = input("Tags (comma-separated): ").strip()
        tags = [tag.strip() for tag in tags_str.split(",")] if tags_str else []
        
        # Get importance score
        importance_str = input("Importance Score (0-1): ").strip()
        importance_score = 0.0
        if importance_str:
            try:
                importance_score = float(importance_str)
                if not 0 <= importance_score <= 1:
                    self.console.print("[bold red]Error:[/bold red] Importance score must be between 0 and 1")
                    return
            except ValueError:
                self.console.print("[bold red]Error:[/bold red] Importance score must be a number")
                return
                
        # Create memory data
        memory_data = {
            "content": content,
            "memory_type": memory_type,
            "tags": tags,
            "importance_score": importance_score,
            "metadata": {
                "source": "memory_cli",
                "created_by": os.getenv("USER", "unknown")
            }
        }
        
        # Add memory
        with Progress() as progress:
            task = progress.add_task("[cyan]Adding memory...", total=1)
            memory_id = await self.memory_handler.add_memory(memory_data)
            progress.update(task, completed=1)
            
        self.console.print(f"[bold green]Memory added successfully![/bold green] ID: {memory_id}")
    
    async def delete_memory(self, args_str: str):
        """Delete a memory by ID"""
        memory_id = args_str.strip()
        if not memory_id:
            self.console.print("[bold red]Error:[/bold red] Memory ID required")
            return
            
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete memory {memory_id}? (y/N): ").strip().lower()
        
        if confirm != "y":
            self.console.print("Deletion cancelled")
            return
            
        # Delete memory
        with Progress() as progress:
            task = progress.add_task("[cyan]Deleting memory...", total=1)
            success = await self.memory_handler.delete_memory(memory_id)
            progress.update(task, completed=1)
            
        if success:
            self.console.print(f"[bold green]Memory deleted successfully![/bold green]")
        else:
            self.console.print(f"[bold red]Error:[/bold red] Memory not found: {memory_id}")
    
    async def update_memory(self, args_str: str):
        """Update a memory by ID"""
        memory_id = args_str.strip()
        if not memory_id:
            self.console.print("[bold red]Error:[/bold red] Memory ID required")
            return
            
        # Get current memory
        table = self.memory_handler.db.open_table("memories")
        existing = table.search().where(f"id = '{memory_id}'").to_list()
        
        if not existing:
            self.console.print(f"[bold red]Error:[/bold red] Memory not found: {memory_id}")
            return
            
        memory = existing[0]
        
        # Display current memory
        self.display_memory_details(memory)
        
        # Get fields to update
        self.console.print("[bold cyan]Update Memory[/bold cyan] (leave empty to keep current value)")
        
        # Get content
        print("Enter new content (press Ctrl+D or Ctrl+Z on Windows when done, or Enter to keep current):")
        content_lines = []
        try:
            while True:
                line = input()
                content_lines.append(line)
        except EOFError:
            content = "\n".join(content_lines).strip()
        
        # Get memory type
        print(f"Available memory types: {', '.join(self.memory_types)}")
        memory_type = input("Memory Type: ").strip()
        
        # Get tags
        current_tags = ", ".join(memory.get("tags", []))
        tags_str = input(f"Tags (current: {current_tags}): ").strip()
        
        # Get importance score
        current_score = memory.get("importance_score", 0)
        importance_str = input(f"Importance Score (current: {current_score}): ").strip()
        
        # Build updates dict
        updates = {}
        if content:
            updates["content"] = content
        if memory_type:
            if memory_type not in self.memory_types:
                self.console.print(f"[bold red]Error:[/bold red] Invalid memory type. Choose from: {', '.join(self.memory_types)}")
                return
            updates["memory_type"] = memory_type
        if tags_str:
            updates["tags"] = [tag.strip() for tag in tags_str.split(",")]
        if importance_str:
            try:
                importance_score = float(importance_str)
                if not 0 <= importance_score <= 1:
                    self.console.print("[bold red]Error:[/bold red] Importance score must be between 0 and 1")
                    return
                updates["importance_score"] = importance_score
            except ValueError:
                self.console.print("[bold red]Error:[/bold red] Importance score must be a number")
                return
                
        if not updates:
            self.console.print("No changes made")
            return
            
        # Update memory
        with Progress() as progress:
            task = progress.add_task("[cyan]Updating memory...", total=1)
            success = await self.memory_handler.update_memory(memory_id, updates)
            progress.update(task, completed=1)
            
        if success:
            self.console.print(f"[bold green]Memory updated successfully![/bold green]")
        else:
            self.console.print(f"[bold red]Error:[/bold red] Failed to update memory")
    
    async def show_tags(self):
        """Show all tags in the database"""
        with Progress() as progress:
            task = progress.add_task("[cyan]Loading tags...", total=1)
            tags = await self.memory_handler.get_all_tags()
            progress.update(task, completed=1)
            
        if not tags:
            self.console.print("[yellow]No tags found[/yellow]")
            return
            
        # Create table
        table = Table(title="All Tags")
        table.add_column("Tag", style="cyan")
        
        for tag in tags:
            table.add_row(tag)
            
        self.console.print(table)
    
    async def show_stats(self):
        """Show database statistics"""
        with Progress() as progress:
            task = progress.add_task("[cyan]Loading statistics...", total=1)
            
            # Get memory counts by type
            stats_query = """
            SELECT 
                memory_type, 
                COUNT(*) as count,
                MIN(created_at) as oldest,
                MAX(created_at) as newest,
                AVG(importance_score) as avg_importance
            FROM memories
            GROUP BY memory_type
            ORDER BY count DESC
            """
            stats = await self.memory_handler.sql_query(stats_query)
            
            # Get total count
            total_query = "SELECT COUNT(*) as total FROM memories"
            total_result = await self.memory_handler.sql_query(total_query)
            total = total_result[0]["total"] if total_result else 0
            
            progress.update(task, completed=1)
            
        # Create table
        table = Table(title=f"Memory Database Statistics (Total: {total})")
        table.add_column("Type", style="green")
        table.add_column("Count", justify="right")
        table.add_column("Oldest", style="dim")
        table.add_column("Newest", style="dim")
        table.add_column("Avg Importance", justify="right")
        
        for stat in stats:
            memory_type = stat["memory_type"]
            count = stat["count"]
            oldest = stat.get("oldest")
            newest = stat.get("newest")
            avg_importance = stat.get("avg_importance", 0)
            
            oldest_str = oldest.strftime("%Y-%m-%d") if oldest else "N/A"
            newest_str = newest.strftime("%Y-%m-%d") if newest else "N/A"
            
            table.add_row(
                memory_type,
                str(count),
                oldest_str,
                newest_str,
                f"{avg_importance:.4f}"
            )
            
        self.console.print(table)


async def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="Memory CLI - A command-line interface for managing LanceDB memories")
    parser.add_argument("--db-path", default="~/.pyasco/memories", help="Path to the LanceDB database")
    
    args = parser.parse_args()
    
    cli = MemoryCLI(db_path=args.db_path)
    await cli.start()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except Exception as e:
        print(f"Error: {e}")
    finally:
        loop.close()
