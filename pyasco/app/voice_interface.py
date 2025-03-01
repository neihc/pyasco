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
import queue
import threading
import time
from typing import Optional
import pyaudio
import wave
from deepgram import Deepgram
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from ..config import ConfigManager
from ..agent import Agent
from ..logger_config import setup_logger

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2.0  # seconds of silence to trigger processing

logger = setup_logger('voice', 'voice.log')
console = Console()

class VoiceInterface:
    def __init__(self, agent: Agent, deepgram_key: str):
        self.agent = agent
        self.deepgram = Deepgram(deepgram_key)
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.last_audio_time = time.time()
        self.console = Console()
        
    async def process_audio(self):
        """Process audio chunks and detect silence"""
        audio_data = []
        silence_start = None
        
        while self.is_recording:
            if not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                audio_data.append(chunk)
                self.last_audio_time = time.time()
                
                # Check audio level for silence
                audio_level = max(abs(float(x)) for x in chunk)
                if audio_level < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        # Process accumulated audio
                        await self.process_speech(b''.join(audio_data))
                        audio_data = []
                        silence_start = None
                else:
                    silence_start = None
                    
            await asyncio.sleep(0.1)
    
    async def process_speech(self, audio_bytes: bytes):
        """Send audio to Deepgram and process transcription"""
        try:
            source = {'buffer': audio_bytes, 'mimetype': 'audio/raw'}
            response = await self.deepgram.transcription.prerecorded(
                source,
                {
                    'punctuate': True,
                    'model': 'general',
                    'language': 'en-US',
                    'encoding': 'linear16',
                    'sample_rate': RATE
                }
            )
            
            transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
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
            logger.error(f"Error processing speech: {str(e)}")
            console.print(f"[red]Error processing speech: {str(e)}[/red]")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    async def start(self):
        """Start voice interface"""
        try:
            # Open audio stream
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=self.audio_callback
            )
            
            console.print("[bold green]Voice interface started! Speak to interact...[/bold green]")
            console.print("[italic](Silence for 2 seconds will trigger processing)[/italic]")
            
            self.is_recording = True
            stream.start_stream()
            
            # Start processing audio
            await self.process_audio()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping voice interface...[/yellow]")
        finally:
            self.is_recording = False
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            self.audio.terminate()
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
