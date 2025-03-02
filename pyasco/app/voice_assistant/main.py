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

import os
import asyncio
import argparse
import logging
import signal
from .assistant import VoiceAssistant
from ...config import ConfigManager
from ...agent import Agent
from ...logger_config import setup_logger
from dotenv import load_dotenv

logger = setup_logger('voice_assistant', 'voice_assistant.log')
load_dotenv()

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
    
    # Initialize voice assistant with proper parameters
    voice_assistant = VoiceAssistant(
        agent=agent,
        voice_id=args.voice_id,
        enable_audio=not args.no_audio
    )
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(voice_assistant, loop)))
    
    logger.info("Starting voice assistant...")
    try:
        await voice_assistant.run()
    except Exception as e:
        logger.error(f"Error in voice assistant: {e}", exc_info=True)
        await shutdown(voice_assistant, loop)

async def shutdown(voice_assistant, loop):
    """Gracefully shutdown the voice assistant"""
    logger.info("Shutting down voice assistant...")
    if hasattr(voice_assistant, 'cleanup'):
        await voice_assistant.cleanup()
    if hasattr(voice_assistant.agent, 'cleanup'):
        voice_assistant.agent.cleanup()
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

if __name__ == "__main__":
    asyncio.run(main())
