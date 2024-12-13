"""
PyAsco Console Application

Usage:
    uv run -m src.app.console [options]
"""

from typing import Optional, Union, List
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from rich.console import Console
from ..config import ConfigManager
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from rich.live import Live
from rich import print as rprint
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from ..agent import Agent

console = Console()

class CommandCompleter(Completer):
    """Completer for magic commands"""
    def __init__(self):
        self.commands = [
            '%exit',
            '%reset',
            '%learn_that_skill',
            '%improve_that_skill'
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

def stream_response(response_generator) -> None:
    """Stream and display response chunks"""
    buffer = ""
    with Live(Markdown(""), refresh_per_second=10) as live:
        for chunk in response_generator:
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
    parser.add_argument("--skills-path", default="skills",
                       help="Path to skills directory")
    return parser.parse_args()

def main():
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
        console.print("  %learn_that_skill - convert current conversation into a reusable skill")
        console.print("  %improve_that_skill - improve an existing skill based on current conversation\n")
        
        loop_count = 0
        user_input = None  # Initialize user_input
        while True:
            if not user_input:  # Only ask for input if we don't have follow-up
                user_input = session.prompt("\nYou> ")
                loop_count = 0  # Reset counter on new user input
                
                if user_input.startswith('%'):
                    command = user_input[1:].lower()
                    if command == 'exit':
                        break
                    elif command == 'reset':
                        agent.reset()
                        console.print("[bold yellow]Chat history reset![/bold yellow]")
                    elif command == 'learn_that_skill':
                        try:
                            skill = agent.learn_that_skill()
                            console.print(f"[bold green]Learned new skill: {skill.name}[/bold green]")
                            console.print(f"Usage: {skill.usage}")
                        except ValueError as e:
                            console.print(f"[bold red]Error learning skill: {str(e)}[/bold red]")
                    elif command == 'improve_that_skill':
                        try:
                            skill = agent.improve_that_skill()
                            console.print(f"[bold green]Improved skill: {skill.name}[/bold green]")
                            console.print(f"New usage: {skill.usage}")
                        except ValueError as e:
                            console.print(f"[bold red]Error improving skill: {str(e)}[/bold red]")
                    user_input = None
                    continue
                
            # Get streaming response
            console.print("\n[bold purple]Assistant[/bold purple]")
            response = agent.get_response(user_input, stream=True)
            stream_response(response)
            
            # Check if we should ask user for code execution
            user_input = None
            if agent.should_ask_user():
                if Confirm.ask("\nDo you want to execute the code snippets?"):
                    results = agent.confirm()
                    if results:
                        console.print("\n[bold yellow]Execution Output:[/bold yellow]")
                        for result in results:
                            console.print(result)
                        
                        user_input = agent.get_follow_up(results)
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
    main()
