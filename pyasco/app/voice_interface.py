"""
PyAsco Voice Interface - An AI-powered Python assistant with voice input

This module provides a voice interface for the PyAsco AI assistant,
using Deepgram for real-time speech-to-text and Rich for terminal output.

Usage:
    python -m pyasco.app.voice_interface [options]

Options:
    --config PATH    Path to YAML configuration file
    All other options from console.py are supported
"""

import argparse
import asyncio
import json
import logging
from dotenv import load_dotenv
import os
import sys
from typing import Optional, Dict, Any
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from ..config import ConfigManager
from ..agent import Agent
from ..logger_config import setup_logger

# Audio parameters
CHANNELS = 1
RATE = 16000

logger = setup_logger('voice', 'voice.log')
console = Console()

class VoiceInterface:
    def __init__(self, agent: Agent, deepgram_key: str):
        self.agent = agent
        # Initialize Deepgram client with API key
        self.deepgram = DeepgramClient()
        self.dg_connection = None
        self.microphone = None
        self.console = Console()
        self.is_listening = True
        logger.info("Voice interface initialized with Deepgram client")

    async def on_message(self, *args, **kwargs):
        """Handle transcription results"""
        try:
            # Extract result from args (first argument)
            result = args[0]
            transcript = result.channel.alternatives[0].transcript
            
            # Only process if we have a non-empty transcript and it's a final result
            if transcript.strip() and not result.is_final:
                logger.debug(f"Interim transcript: {transcript}")
                console.print(f"[dim italic]Hearing: {transcript}[/dim italic]", end="\r")
            
            elif transcript.strip() and result.is_final:
                logger.info(f"Final transcript: {transcript}")
                console.print(f"\n[bold yellow]You said: {transcript}[/bold yellow]")
                
                with console.status("[bold green]Processing your request..."):
                    response = await self.agent.ask(transcript, stream=True)
                    
                    # Display streaming response
                    with Live(auto_refresh=False) as live:
                        async for chunk in response:
                            panel = Panel(
                                Text(chunk.content, style="bold blue"),
                                title="AI Response",
                                border_style="green"
                            )
                            live.update(panel, refresh=True)
                
                console.print("\n[italic]Listening for your next question...[/italic]")
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}", exc_info=True)
            console.print(f"[red]Error processing transcript: {str(e)}[/red]")

    def on_metadata(self, *args, **kwargs):
        """Handle metadata events"""
        logger.debug(f"Metadata received: {args}")
        # Show a visual indicator that the system is receiving audio
        console.print("[dim].", end="")

    def on_error(self, *args, **kwargs):
        """Handle error events"""
        error_msg = args[0] if args else "Unknown error"
        logger.error(f"Deepgram error: {error_msg}")
        console.print(f"\n[red]Deepgram error: {error_msg}[/red]")
        
    def on_close(self, *args, **kwargs):
        """Handle connection close events"""
        logger.info("Deepgram connection closed")
        console.print("\n[yellow]Deepgram connection closed[/yellow]")

    async def start(self):
        """Start voice interface"""
        try:
            logger.info("Starting voice interface")
            # Setup Deepgram connection
            self.dg_connection = self.deepgram.listen.live.v("1")
            
            # Configure event handlers
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Metadata, self.on_metadata)
            self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)
            self.dg_connection.on(LiveTranscriptionEvents.Close, self.on_close)
            
            # Configure transcription options
            options = LiveOptions(
                model="nova-2",
                punctuate=True,
                language="en-US",
                encoding="linear16",
                channels=CHANNELS,
                sample_rate=RATE,
                interim_results=True,
                utterance_end_ms="1000",
                smart_format=True,
            )
            
            # Start the connection
            logger.info("Starting Deepgram connection")
            self.dg_connection.start(options)
            
            # Setup and start microphone
            logger.info("Starting microphone")
            self.microphone = Microphone(self.dg_connection.send)
            self.microphone.start()
            
            console.print("[bold green]Voice interface started! Speak to interact...[/bold green]")
            console.print("[italic](Press Enter to stop)[/italic]")
            
            # Create a way to exit the program with Enter key
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            
            # Add a reader to detect when Enter is pressed
            loop.add_reader(sys.stdin, lambda: future.set_result(None) if not future.done() else None)
            
            # Wait for user to press Enter
            await future
            console.print("\n[yellow]Stopping voice interface...[/yellow]")
            
        except Exception as e:
            logger.error(f"Error in voice interface: {str(e)}", exc_info=True)
            console.print(f"\n[red]Error: {str(e)}[/red]")
        finally:
            logger.info("Cleaning up resources")
            if self.microphone:
                self.microphone.finish()
            if self.dg_connection:
                self.dg_connection.finish()
            self.agent.cleanup()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PyAsco Voice Interface")
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--deepgram-key",
                       help="Deepgram API key (can also be set via DEEPGRAM_API_KEY env var)")
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct",
                       help="LLM model to use for responses")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level")
    return parser.parse_args()

async def main():
    """Main function to run the voice interface"""
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger('voice_interface', log_file='voice.log', level=log_level)
    
    # Load environment variables
    load_dotenv()
    
    # Get Deepgram key from args or environment
    deepgram_key = args.deepgram_key or os.getenv('DEEPGRAM_API_KEY')
    if not deepgram_key:
        console.print("[bold red]Deepgram API key is missing![/bold red]")
        console.print("Provide via --deepgram-key or DEEPGRAM_API_KEY env var")
        return
    
    logger.info(f"Starting voice interface with log level: {args.log_level}")
        
    # Load configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        config = ConfigManager.load_from_yaml(args.config)
    else:
        logger.info("Loading configuration from command line arguments")
        config = ConfigManager.from_args(args)
    
    # Initialize agent and interface
    logger.info("Initializing agent")
    agent = Agent(config)
    
    logger.info("Initializing voice interface")
    interface = VoiceInterface(agent, deepgram_key)
    
    try:
        logger.info("Starting voice interface")
        await interface.start()
    except Exception as e:
        logger.error(f"Error in voice interface: {str(e)}", exc_info=True)
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
    finally:
        logger.info("Cleaning up")
        agent.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Voice interface stopped by user[/red]")
