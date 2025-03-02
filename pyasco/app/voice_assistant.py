"""
PyAsco Voice Assistant - An AI-powered Python assistant with voice input and output

This module provides a voice interface for the PyAsco AI assistant,
capable of listening to voice input, displaying responses in real-time,
and speaking responses using ElevenLabs text-to-speech.

Usage:
    python -m pyasco.app.voice_assistant [options]

Options:
    --config PATH          Path to YAML configuration file
    --model TEXT           LLM model to use
    --log-level TEXT       Logging level (DEBUG, INFO, WARNING, ERROR)
    --voice-id TEXT        ElevenLabs voice ID to use for speech output
    --no-audio             Disable audio output (text-only mode)
    All other options from console.py are supported
"""

import argparse
import asyncio
import os
import sys
import logging
import time
import json
import base64
import shutil
import subprocess
import websockets
from typing import List, Dict, Optional, Any, AsyncGenerator, Iterator
from asyncio import Task
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
    def __init__(self, max_sentences=10):
        self.reset()
        self.full_sentences = []
        self.current_partial = ""
        self.max_sentences = max_sentences
        
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
            # Keep only the last max_sentences
            if len(self.full_sentences) > self.max_sentences:
                self.full_sentences = self.full_sentences[-self.max_sentences:]
    
    def get_combined_input(self):
        """Get combined input from the sliding window of sentences"""
        return ' '.join(self.full_sentences)
            
    def get_display_text(self):
        """Get text for display purposes"""
        # Return the last few sentences and current partial
        history = self.full_sentences[-3:] if self.full_sentences else []
        if self.current_partial:
            return "\n".join(history + [f"🎤 {self.current_partial}"])
        else:
            return "\n".join(history + ["🎤 Listening..."])

class VoiceAssistant:
    """Voice interface for PyAsco AI assistant with speech output"""
    def __init__(self, agent: Agent, voice_id: Optional[str] = None, enable_audio: bool = True):
        self.agent = agent
        self.transcript_collector = TranscriptCollector(max_sentences=10)
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.enable_audio = enable_audio and bool(self.elevenlabs_api_key)
        self.current_task: Optional[Task] = None
        self.response_text = ""
        self.microphone = None
        self.dg_connection = None
        self.live = None  # Store the live display object
        self.tts_task = None  # Task for text-to-speech streaming
        
        # UI components
        self.console = Console()
        self.layout = Layout()
        self._setup_layout()
        
        # Check for mpv if audio is enabled
        if self.enable_audio and not self._is_installed("mpv"):
            logger.warning("mpv not found, audio output disabled. Install mpv to enable audio: https://mpv.io/installation/")
            self.enable_audio = False
        
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
        
    def _cancel_current_task(self):
        """Cancel the current processing task if it exists"""
        if self.current_task and not self.current_task.done():
            logger.debug("Cancelling current processing task")
            self.current_task.cancel()
            
    async def _process_voice_input_task(self, text: str):
        """Background task to process voice input with the agent"""
        logger.debug(f"Processing voice input in background task: '{text}'")
        try:
            # Show processing status
            self.response_text = "Processing..."
            self._update_display()
            
            # Get and display streaming response
            await self._get_agent_response(text)
            
            # Handle code execution if needed
            await self._handle_code_execution()
            
        except asyncio.CancelledError:
            logger.info("Voice processing task was cancelled")
            await self.agent.stop_stream()
            raise
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}", exc_info=True)
            self.response_text = f"Error: {str(e)}"
            self._update_display()
            
    async def _get_agent_response(self, text: str, new_session=True):
        """Get response from agent and handle streaming"""
        try:
            logger.debug("Sending request to agent")
            response = await self.agent.ask(text, new_session=new_session, stream=True)
            
            # Handle streaming response
            self.response_text = ""
            logger.debug("Processing streaming response")
            
            # Create text iterator for TTS
            if self.enable_audio:
                # Cancel any existing TTS task
                if self.tts_task and not self.tts_task.done():
                    self.tts_task.cancel()
                
                # Start TTS streaming in background
                text_chunks = self._create_text_chunk_generator(response)
                self.tts_task = asyncio.create_task(
                    self._stream_text_to_speech(text_chunks)
                )
                
                # Process the same response for display
                for chunk in response:
                    if chunk.content:
                        self.response_text += chunk.content
                        self._update_display()
                        await asyncio.sleep(0.05)  # Small delay to reduce update frequency
            else:
                # Process response without TTS
                for chunk in response:
                    if chunk.content:
                        self.response_text += chunk.content
                        self._update_display()
                        await asyncio.sleep(0.05)  # Small delay to reduce update frequency
                        
        except Exception as e:
            logger.error(f"Error getting response from agent: {str(e)}", exc_info=True)
            self.response_text = f"Error: {str(e)}"
            self._update_display()
            
    async def _handle_code_execution(self):
        """Handle code execution and follow-ups"""
        should_execute = await self.agent.should_ask_user()
        logger.debug(f"Should execute code: {should_execute}")
        
        if not should_execute:
            return
            
        max_loops = 5
        loop_count = 0
        
        while await self.agent.should_ask_user() and not self.agent.should_stop_follow_up(loop_count, max_loops):
            self.response_text = "\n\n*Executing code...*"
            self._update_display()
            logger.debug(f"Auto-executing code (loop {loop_count+1}/{max_loops})")
            
            # Execute current tools
            results = self.agent.confirm()
            if not results:
                logger.debug("No results from execution")
                break
                
            # Display execution results
            self._display_execution_results(results)
            
            # Get follow-up if needed
            await self._process_follow_up(results)
            await asyncio.sleep(0)
            
            loop_count += 1
        
        if loop_count >= max_loops:
            logger.warning("Reached maximum follow-up iterations")
            self.response_text = "\n\n*Reached maximum number of execution steps*"
            
    def _display_execution_results(self, results):
        """Display execution results in the UI"""
        logger.debug(f"Execution results: {len(results)} items")
        result_text = "\n\n**Execution Results:**\n```\n"
        result_text += "\n".join(results)
        result_text += "\n```"
        self.response_text = result_text
        self._update_display()
        
    async def _process_follow_up(self, results):
        """Process follow-up queries based on execution results"""
        logger.debug("Getting follow-up")
        follow_up = await self.agent.get_follow_up(results)
        if follow_up:
            logger.debug(f"Follow-up query: '{follow_up}'")
            await self._get_agent_response(follow_up, new_session=False)
    
    async def process_voice_input(self, text):
        """Process voice input with the agent"""
        if not text.strip():
            logger.debug("Skipping processing: empty input")
            return
            
        # Cancel any existing task and stop ongoing streams
        self._cancel_current_task()
        await self.agent.stop_stream()
        
        # Start a new background task
        logger.debug(f"Creating new task for input: '{text}'")
        self.current_task = asyncio.create_task(self._process_voice_input_task(text))
            
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
                
                # Update the display
                self._update_display()
                
                # Process the combined sentences - don't await to avoid blocking
                combined_input = self.transcript_collector.get_combined_input()
                logger.debug(f"Processing combined input from {len(self.transcript_collector.full_sentences)} sentences")
                asyncio.create_task(self.process_voice_input(combined_input))
                
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
        
    def _is_installed(self, lib_name):
        """Check if a system library is installed"""
        return shutil.which(lib_name) is not None
        
    def _create_text_chunk_generator(self, response_generator):
        """Create a generator that yields text chunks from the response"""
        async def text_iterator():
            for chunk in response_generator:
                if chunk.content:
                    yield chunk.content
        return text_iterator()
    
    async def _text_chunker(self, chunks):
        """Split text into chunks, ensuring to not break sentences."""
        splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
        buffer = ""

        async for text in chunks:
            if not text:
                continue
                
            if buffer.endswith(splitters):
                yield buffer + " "
                buffer = text
            elif text.startswith(splitters):
                yield buffer + text[0] + " "
                buffer = text[1:]
            else:
                buffer += text

        if buffer:
            yield buffer + " "
    
    async def _stream_audio(self, audio_stream):
        """Stream audio data using mpv player."""
        if not self._is_installed("mpv"):
            logger.error("mpv not found, necessary to stream audio")
            return

        mpv_process = subprocess.Popen(
            ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        logger.debug("Started streaming audio")
        try:
            async for chunk in audio_stream:
                if chunk and mpv_process.stdin:
                    mpv_process.stdin.write(chunk)
                    mpv_process.stdin.flush()
        except Exception as e:
            logger.error(f"Error streaming audio: {str(e)}")
        finally:
            if mpv_process.stdin:
                mpv_process.stdin.close()
            mpv_process.wait()
    
    async def _stream_text_to_speech(self, text_iterator):
        """Send text to ElevenLabs API and stream the returned audio."""
        if not self.elevenlabs_api_key:
            logger.error("ElevenLabs API key not found")
            return
            
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id=eleven_flash_v2_5"
        
        try:
            logger.debug(f"Connecting to ElevenLabs with voice ID: {self.voice_id}")
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps({
                    "text": " ",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                    "xi_api_key": self.elevenlabs_api_key,
                }))

                async def listen():
                    """Listen to the websocket for audio data and stream it."""
                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)
                            if data.get("audio"):
                                yield base64.b64decode(data["audio"])
                            elif data.get('isFinal'):
                                break
                        except websockets.exceptions.ConnectionClosed:
                            logger.debug("ElevenLabs connection closed")
                            break

                listen_task = asyncio.create_task(self._stream_audio(listen()))

                async for text in self._text_chunker(text_iterator):
                    await websocket.send(json.dumps({"text": text}))

                await websocket.send(json.dumps({"text": ""}))
                await listen_task
                
        except Exception as e:
            logger.error(f"Error in text-to-speech streaming: {str(e)}")
    
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
            with Live(self.layout, refresh_per_second=4, auto_refresh=True) as live:
                self.live = live  # Store the live object
                status = "[bold green]Voice Assistant started. Speak to interact!"
                if self.enable_audio:
                    status += " [bold blue](Audio output enabled)[/]"
                else:
                    status += " [bold yellow](Audio output disabled)[/]"
                self.console.print(status)
                self._update_display()  # Initial display update
                
                # Main loop
                logger.debug("Entering main loop")
                while True:
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
            self._cancel_current_task()
            # Cancel TTS task if running
            if self.tts_task and not self.tts_task.done():
                self.tts_task.cancel()
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
    parser.add_argument("--voice-id", 
                       help="ElevenLabs voice ID to use for speech output")
    parser.add_argument("--no-audio", action="store_true",
                       help="Disable audio output (text-only mode)")
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
    voice_assistant = VoiceAssistant(
        agent,
        voice_id=args.voice_id,
        enable_audio=not args.no_audio
    )
    
    # Run the voice assistant
    await voice_assistant.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nVoice assistant stopped by user")
