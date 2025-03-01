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
from typing import Optional
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
        self.deepgram = DeepgramClient(
            DeepgramClientOptions(api_key=deepgram_key)
        )
        self.dg_connection = None
        self.microphone = None
        self.console = Console()

    async def on_message(self, *args, **kwargs):
        """Handle transcription results"""
        try:
            # Extract result from args (first argument)
            result = args[0]
            transcript = result.channel.alternatives[0].transcript
            if transcript.strip():
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
        except Exception as e:
            logger.error(f"Error processing transcript: {str(e)}")
            console.print(f"[red]Error processing transcript: {str(e)}[/red]")

    def on_metadata(self, *args, **kwargs):
        """Handle metadata events"""
        logger.debug(f"Metadata received: {args}")

    def on_error(self, *args, **kwargs):
        """Handle error events"""
        logger.error(f"Deepgram error: {args}")
        console.print(f"[red]Deepgram error: {args}[/red]")

    async def start(self):
        """Start voice interface"""
        try:
            # Setup Deepgram connection
            self.dg_connection = self.deepgram.listen.live.v("1")
            
            # Configure event handlers
            self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
            self.dg_connection.on(LiveTranscriptionEvents.Metadata, self.on_metadata)
            self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)
            
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
            )
            
            # Start the connection
            self.dg_connection.start(options)
            
            # Setup and start microphone
            self.microphone = Microphone(self.dg_connection.send)
            self.microphone.start()
            
            console.print("[bold green]Voice interface started! Speak to interact...[/bold green]")
            console.print("[italic](Press Enter to stop)[/italic]")
            
            # Wait for user to stop
            await asyncio.Event().wait()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping voice interface...[/yellow]")
        finally:
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
    logger = setup_logger(__name__, log_file='voice.log', level=log_level)
    
    # Load environment variables
    load_dotenv()
    
    # Get Deepgram key from args or environment
    deepgram_key = args.deepgram_key or os.getenv('DEEPGRAM_API_KEY')
    if not deepgram_key:
        logger.error("Deepgram API key is missing! Provide via --deepgram-key or DEEPGRAM_API_KEY env var")
        return
        
    # Load configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        config = ConfigManager.load_from_yaml(args.config)
    else:
        logger.info("Loading configuration from command line arguments")
        config = ConfigManager.from_args(args)
    
    # Initialize agent and interface
    agent = Agent(config)
    interface = VoiceInterface(agent, deepgram_key)
    
    try:
        await interface.start()
    except Exception as e:
        logger.error(f"Error in voice interface: {str(e)}", exc_info=True)
    finally:
        agent.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Voice interface stopped by user[/red]")
