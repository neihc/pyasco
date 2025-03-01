"""
PyAsco Voice Interface - An AI-powered Python assistant with voice interaction

This module provides a voice interface for the PyAsco AI assistant,
using Deepgram for speech-to-text and system TTS for text-to-speech.

Usage:
    python -m pyasco.app.voice_interface [options]

Options:
    --config PATH          Path to YAML configuration file
    --deepgram-key TEXT   Deepgram API key for speech recognition
    All other options from console.py are supported
"""

import argparse
import os
import logging
import queue
import threading
import time
import json
import pyaudio
import wave
from typing import Optional, Dict
from pathlib import Path
import platform

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)

from ..config import ConfigManager
from ..agent import Agent
from ..logger_config import setup_logger

# Audio recording parameters
CHUNK = 80000  # 5 seconds of audio at 16kHz
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

logger = setup_logger('voice', 'voice.log')

class VoiceInterface:
    def __init__(self, agent: Agent, deepgram_key: str):
        self.agent = agent
        self.deepgram = DeepgramClient(deepgram_key)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # Initialize text-to-speech based on platform
        if platform.system() == 'Darwin':  # macOS
            import subprocess
            self.tts = lambda text: subprocess.run(['say', text])
        else:  # Linux and others - use espeak
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts = lambda text: self.tts_engine.say(text)

    def _setup_audio_stream(self):
        """Setup audio input stream"""
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

    async def process_audio(self):
        """Process audio stream with Deepgram"""
        logger.info("Setting up Deepgram connection...")
        
        # Create websocket connection
        dg_connection = self.deepgram.listen.live.v("1")
        
        # Configure options
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
        )

        # Define event handlers
        async def on_message(transcript_result, **kwargs):
            try:
                if transcript_result.is_final:
                    transcript = transcript_result.channel.alternatives[0].transcript
                    if transcript.strip():
                        logger.info(f"Recognized: {transcript}")
                        await self.process_text(transcript)
            except Exception as e:
                logger.error(f"Error in message handler: {str(e)}")

        async def on_error(error, **kwargs):
            logger.error(f"Deepgram error: {error}")

        async def on_close():
            logger.info("Deepgram connection closed")

        # Register handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        logger.info("Starting Deepgram connection...")
        dg_connection.start(options)
        logger.info("Successfully connected to Deepgram")

        # Stream audio data
        while self.is_recording:
            try:
                if not self.audio_queue.empty():
                    data = self.audio_queue.get()
                    if data: # Ensure we have valid data
                        dg_connection.send(data)
                        logger.debug(f"Sent audio chunk to Deepgram")
                await asyncio.sleep(0.01)  # Shorter sleep to be more responsive
            except Exception as e:
                logger.error(f"Error streaming audio: {str(e)}")
                if "not connected" in str(e).lower():
                    logger.info("Reconnecting to Deepgram...")
                    dg_connection.start(options)
                else:
                    break

        # Close connection
        logger.info("Closing Deepgram connection...")
        dg_connection.finish()


    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self._setup_audio_stream()
        
        def audio_callback():
            while self.is_recording:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except Exception as e:
                    logger.error(f"Error reading audio: {str(e)}")
                    break

        self.audio_thread = threading.Thread(target=audio_callback)
        self.audio_thread.start()

    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio_thread:
            self.audio_thread.join()

    async def process_text(self, text: str):
        """Process transcribed text with the agent"""
        try:
            # Speak processing message
            self.tts("Processing...")
            
            # Get response from agent
            response = await self.agent.ask(text, stream=False)
            
            # Speak the response
            self.tts(response.content)
            
            # Handle any code execution
            if await self.agent.should_ask_user():
                self.tts("Would you like me to execute the code? Say yes or no.")
                # Note: In a full implementation, you'd want to listen for the response
                # and handle the confirmation flow
        
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            self.tts("Sorry, I encountered an error processing your request.")

    def cleanup(self):
        """Cleanup resources"""
        self.stop_recording()
        self.audio.terminate()
        if hasattr(self, 'tts_engine'):
            self.tts_engine.stop()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PyAsco Voice Interface")
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--deepgram-key", required=True,
                       help="Deepgram API key")
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
    
    if not args.deepgram_key:
        logger.error("Deepgram API key is missing!")
        return
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        config = ConfigManager.load_from_yaml(args.config)
    else:
        logger.info("Loading configuration from command line arguments")
        config = ConfigManager.from_args(args)
    
    # Initialize agent and interface
    logger.info("Initializing agent and voice interface...")
    agent = Agent(config)
    interface = VoiceInterface(agent, args.deepgram_key)
    
    # Start recording and processing
    logger.info("Starting voice interface...")
    print("Voice interface started! Speak to interact. Press Ctrl+C to stop.")
    interface.start_recording()
    await interface.process_audio()
    interface.cleanup()
    agent.cleanup()

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nVoice interface stopped by user")
