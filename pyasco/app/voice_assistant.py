"""
PyAsco Voice Assistant - An AI-powered Python assistant with voice input

This module provides a voice interface for the PyAsco AI assistant,
capable of listening to voice input and displaying responses in real-time.

Usage:
    python -m pyasco.app.voice_assistant [options]

Options:
    --config PATH          Path to YAML configuration file
    --model TEXT           LLM model to use
    --log-level TEXT       Logging level (DEBUG, INFO, WARNING, ERROR)
    All other options from console.py are supported
"""

import argparse
import asyncio
import os
import sys
import logging
import time
from typing import List, Dict, Optional, Any
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich import box
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

from ..config import ConfigManager, Config
from ..agent import Agent
from ..logger_config import setup_logger

# Load environment variables
load_dotenv()

# Setup logging
logger = setup_logger('voice_assistant', 'voice_assistant.log', level=logging.DEBUG)
console = Console()

class TranscriptCollector:
    """Collects and manages transcript parts from Deepgram"""
    def __init__(self):
        self.reset()
        self.full_sentences = []
        self.current_partial = ""
        
    def reset(self):
        """Reset the current transcript collection"""
        self.transcript_parts = []
        
    def add_part(self, part):
        """Add a transcript part"""
        self.transcript_parts.append(part)
        
    def get_full_transcript(self):
        """Get the full transcript from collected parts"""
        return ' '.join(self.transcript_parts)
    
    def update_partial(self, text):
        """Update the current partial transcript"""
        self.current_partial = text
        
    def add_sentence(self, sentence):
        """Add a completed sentence to the history"""
        if sentence.strip():
            self.full_sentences.append(sentence.strip())
            
    def get_display_text(self):
        """Get text for display purposes"""
        # Return the last few sentences and current partial
        history = self.full_sentences[-3:] if self.full_sentences else []
        if self.current_partial:
            return "\n".join(history + [f"🎤 {self.current_partial}"])
        else:
            return "\n".join(history + ["🎤 Listening..."])

class VoiceAssistant:
    """Voice interface for PyAsco AI assistant"""
    def __init__(self, agent: Agent):
        self.agent = agent
        self.transcript_collector = TranscriptCollector()
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "")
        self.is_processing = False
        self.response_text = ""
        self.microphone = None
        self.dg_connection = None
        self.live = None  # Store the live display object
        
        # UI components
        self.console = Console()
        self.layout = Layout()
        self._setup_layout()
        
    def _setup_layout(self):
        """Setup the rich layout for the UI"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1)
        )
        
        self.layout["main"].split_row(
            Layout(name="input", ratio=1),
            Layout(name="response", ratio=1),
        )
        
    def _update_display(self):
        """Update the display with current state"""
        if not self.live:
            return
            
        # Header
        self.layout["header"].update(
            Panel(
                Text("PyAsco Voice Assistant", style="bold blue", justify="center"),
                box=box.ROUNDED,
                border_style="blue"
            )
        )
        
        # Input panel
        self.layout["input"].update(
            Panel(
                Text(self.transcript_collector.get_display_text()),
                title="Voice Input",
                border_style="green",
                box=box.ROUNDED
            )
        )
        
        # Response panel
        self.layout["response"].update(
            Panel(
                Markdown(self.response_text) if self.response_text else Text("Waiting for input..."),
                title="AI Response",
                border_style="yellow",
                box=box.ROUNDED
            )
        )
        
        # Render the layout
        self.live.update(self.layout)
        
    async def process_voice_input(self, text):
        """Process voice input with the agent"""
        if not text.strip() or self.is_processing:
            logger.debug(f"Skipping processing: empty={not text.strip()}, already_processing={self.is_processing}")
            return
            
        self.is_processing = True
        logger.debug(f"Processing voice input: '{text}'")
        try:
            # Clear previous response
            self.response_text = "Processing..."
            
            # Get response from agent
            logger.debug("Sending request to agent")
            response = await self.agent.ask(text, stream=True)
            
            # Handle streaming response
            self.response_text = ""
            logger.debug("Processing streaming response")
            for chunk in response:
                if chunk.content:
                    self.response_text += chunk.content
                    # Update the display with each chunk
                    self._update_display()
            
            # Execute any code if needed
            should_execute = await self.agent.should_ask_user()
            logger.debug(f"Should execute code: {should_execute}")
            if should_execute:
                self.response_text += "\n\n*Executing code...*"
                logger.debug("Confirming code execution")
                results = self.agent.confirm()
                if results:
                    logger.debug(f"Execution results: {len(results)} items")
                    result_text = "\n\n**Execution Results:**\n```\n"
                    result_text += "\n".join(results)
                    result_text += "\n```"
                    self.response_text += result_text
                    
                    # Get follow-up if needed
                    logger.debug("Getting follow-up")
                    follow_up = self.agent.get_follow_up(results)
                    if follow_up:
                        logger.debug(f"Follow-up query: '{follow_up}'")
                        follow_up_response = await self.agent.get_response(follow_up, stream=False)
                        logger.debug(f"Follow-up response received: {len(follow_up_response.content)} chars")
                        self.response_text += "\n\n" + follow_up_response.content
        
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            self.response_text = f"Error: {str(e)}"
        finally:
            self.is_processing = False
            
    async def setup_deepgram(self):
        """Setup Deepgram connection"""
        logger.debug("Setting up Deepgram connection")
        if not self.deepgram_api_key:
            logger.error("Deepgram API key not found")
            raise ValueError("Deepgram API key not found. Please set DEEPGRAM_API_KEY in your .env file.")
            
        config = DeepgramClientOptions(options={"keepalive": "true"})
        logger.debug("Initializing Deepgram client")
        deepgram = DeepgramClient(self.deepgram_api_key, config)
        
        logger.debug("Creating Deepgram live connection")
        self.dg_connection = deepgram.listen.asynclive.v("1")
        
        # Define event handlers
        async def on_message(this, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not sentence.strip():
                return
            
            logger.debug(f"Deepgram transcript: '{sentence}', final: {result.speech_final}")
                
            if not result.speech_final:
                # Update the partial transcript
                self.transcript_collector.update_partial(sentence)
            else:
                # This is the final part of the current sentence
                self.transcript_collector.add_part(sentence)
                full_sentence = self.transcript_collector.get_full_transcript()
                logger.debug(f"Final sentence: '{full_sentence}'")
                
                # Add to completed sentences
                self.transcript_collector.add_sentence(full_sentence)
                self.transcript_collector.update_partial("")
                
                # Process the completed sentence
                logger.debug("Processing completed sentence")
                await self.process_voice_input(full_sentence)
                
                # Reset for next sentence
                self.transcript_collector.reset()
        
        async def on_error(this, error, **kwargs):
            logger.error(f"Deepgram error: {error}")
            
        # Register event handlers
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        
        # Configure options
        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=True
        )
        
        logger.debug("Starting Deepgram connection with options")
        await self.dg_connection.start(options)
        logger.debug("Deepgram connection started successfully")
        
    async def run(self):
        """Run the voice assistant"""
        try:
            logger.info("Starting voice assistant")
            # Setup Deepgram
            await self.setup_deepgram()
            
            # Open microphone stream
            logger.debug("Opening microphone stream")
            self.microphone = Microphone(self.dg_connection.send)
            self.microphone.start()
            logger.info("Microphone activated and listening")
            
            # Display UI
            logger.debug("Setting up live display")
            with Live(self.layout, refresh_per_second=4) as live:
                self.live = live  # Store the live object
                self.console.print("[bold green]Voice Assistant started. Speak to interact![/]")
                
                # Main loop
                logger.debug("Entering main loop")
                while True:
                    self._update_display()
                    await asyncio.sleep(0.25)
                    
                    if not self.microphone.is_active():
                        logger.warning("Microphone is no longer active")
                        break
        
        except KeyboardInterrupt:
            self.console.print("[bold yellow]Stopping voice assistant...[/]")
        except Exception as e:
            logger.error(f"Error in voice assistant: {str(e)}", exc_info=True)
            self.console.print(f"[bold red]Error: {str(e)}[/]")
        finally:
            # Cleanup
            if self.microphone:
                self.microphone.finish()
            if self.dg_connection:
                self.dg_connection.finish()
            self.agent.cleanup()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PyAsco Voice Assistant")
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct",
                       help="LLM model to use for responses")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level")
    return parser.parse_args()

async def main():
    """Main function to run the voice assistant"""
    args = parse_args()
    
    # Setup logging with command line specified level
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger(__name__, log_file='voice_assistant.log', level=log_level)
    logger.info("Voice assistant starting up")
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        config = ConfigManager.load_from_yaml(args.config)
    else:
        logger.info("Loading configuration from command line arguments")
        config = ConfigManager.from_args(args)
    
    # Initialize agent
    logger.info("Initializing agent...")
    agent = Agent(config)
    
    # Initialize voice assistant
    voice_assistant = VoiceAssistant(agent)
    
    # Run the voice assistant
    await voice_assistant.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nVoice assistant stopped by user")
