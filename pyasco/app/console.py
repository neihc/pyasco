"""
PyAsco Console Application - An AI-powered Python assistant

This module provides an interactive console interface for the PyAsco AI assistant,
capable of executing Python code, managing skills, and providing intelligent responses.

Usage:
    python -m src.app.console [options]

Options:
    --config PATH          Path to YAML configuration file
    --use-docker          Run code in Docker environment
    --docker-image TEXT   Docker image to use (default: python:3.9-slim)
    --mem-limit TEXT      Docker memory limit (default: 512m)
    --cpu-count INTEGER   Docker CPU count (default: 1)
    --env-file PATH       Path to environment file for Docker
    --mount TEXT          Mount points in format 'host_path:container_path'
    --model TEXT          LLM model to use (default: meta-llama/llama-3.3-70b-instruct)
    --skills-path TEXT    Path to skills directory (default: skills)
"""

from typing import Optional, Union, List, AsyncGenerator
import argparse
import os
import warnings
import asyncio
warnings.filterwarnings("ignore")
from rich.console import Console
from ..config import ConfigManager
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from rich.live import Live
from rich import print as rprint
from prompt_toolkit.shortcuts.prompt import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.application.current import get_app
from ..agent import Agent

console = Console()

class CommandCompleter(Completer):
    """Completer for magic commands"""
    def __init__(self):
        self.commands = [
            '%exit',
            '%reset',
            '%remember'
        ]
    
    def get_completions(self, document, complete_event):
        word = document.get_word_before_cursor()
        
        # If word already starts with %, don't add another one
        has_prefix = word.startswith('%')
        search_word = word[1:] if has_prefix else word
        
        for command in self.commands:
            cmd_without_prefix = command[1:]  # Remove % for comparison
            
            if search_word.lower() in cmd_without_prefix.lower():
                # If word already has %, complete without %, otherwise add %
                completion = cmd_without_prefix if has_prefix else command
                yield Completion(
                    completion,
                    start_position=-len(word)-1
                )

# Initialize prompt session with history
session = PromptSession(
    history=FileHistory('.pyasco_history'),
    completer=CommandCompleter(),
    style=Style.from_dict({
        'prompt': 'bold green',
    })
)

def display_markdown(text: str) -> None:
    """Display text as markdown"""
    md = Markdown(text)
    console.print(text)

async def stream_response(response_generator: AsyncGenerator) -> None:
    """Stream and display response chunks"""
    buffer = ""
    with Live(Markdown(""), refresh_per_second=10) as live:
        async for chunk in response_generator:
            if chunk.content:
                buffer += chunk.content
                live.update(Markdown(buffer))

# Maximum number of follow-up iterations
MAX_FOLLOW_UP_LOOPS = 5

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PyAsco Chat Console")
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--use-docker", action="store_true", help="Run code in Docker")
    parser.add_argument("--docker-image", default="python:3.9-slim", help="Docker image to use")
    parser.add_argument("--mem-limit", default="512m", help="Docker memory limit (e.g., 512m, 1g)")
    parser.add_argument("--cpu-count", type=int, default=1, help="Docker CPU count")
    parser.add_argument("--env-file", help="Path to environment file for Docker container")
    parser.add_argument("--mount", action='append', 
                       help="Mount points in format 'host_path:container_path'. Can be specified multiple times")
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct",
                       help="LLM model to use for responses")
    parser.add_argument("--llm-base-url", default="https://openrouter.ai/api/v1",
                       help="Base URL for LLM API")
    parser.add_argument("--llm-api-key",
                       help="API key for LLM service (defaults to OPENROUTER_API_KEY env var)")
    parser.add_argument("--skills-path", default="skills",
                       help="Path to skills directory")
    return parser.parse_args()

async def main():
    """Main console application loop"""
    args = parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = ConfigManager.load_from_yaml(args.config)
    else:
        config = ConfigManager.from_args(args)
    
    # Initialize agent with configuration
    agent = Agent(config)
    
    try:
        console.print("\n[bold blue]Welcome to PyAsco Chat![/bold blue]")
        if config.docker.use_docker:
            console.print(f"[bold green]Running code in Docker ({config.docker.image})[/bold green]")
            if config.docker.env_file:
                console.print(f"[bold green]Environment loaded from: {config.docker.env_file}[/bold green]")
        console.print("Magic commands:")
        console.print("  %exit - quit the console")
        console.print("  %reset - start over")
        console.print("  %remember - store current conversation in memory\n")
        
        loop_count = 0
        user_input = None  # Initialize user_input
        recall = True
        while True:
            if not user_input:  # Only ask for input if we don't have follow-up
                user_input = await session.prompt_async("\nYou> ")
                loop_count = 0  # Reset counter on new user input
                
                if user_input.startswith('%'):
                    command = user_input[1:].lower()
                    if command == 'exit':
                        break
                    elif command == 'reset':
                        agent.reset()
                        console.print("[bold yellow]Chat history reset![/bold yellow]")
                    elif command == 'remember':
                        try:
                            agent.remember_conversation()
                            console.print("[bold green]Conversation stored in memory![/bold green]")
                        except Exception as e:
                            console.print(f"[bold red]Error storing conversation: {str(e)}[/bold red]")
                    user_input = None
                    continue
                
            # Get streaming response
            console.print("\n[bold purple]Assistant[/bold purple]")
            response = await agent.ask(user_input, recall=recall, stream=True)
            await stream_response(response)
            
            # Check if we should ask user for code execution
            user_input = None
            recall = True
            if agent.should_ask_user():
                if Confirm.ask("\nDo you want to execute the code snippets?"):
                    results = agent.confirm()
                    if results:
                        console.print("\n[bold yellow]Execution Output:[/bold yellow]")
                        for result in results:
                            console.print(result)
                        
                        user_input = agent.get_follow_up(results)
                        recall = False
                        loop_count += 1
                        if agent.should_stop_follow_up(loop_count, MAX_FOLLOW_UP_LOOPS):
                            console.print("\n[bold yellow]Maximum follow-up iterations reached![/bold yellow]")
                            break
                        continue

    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting...[/bold red]")
    finally:
        agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
