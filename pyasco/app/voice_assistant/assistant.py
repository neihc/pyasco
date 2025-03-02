import asyncio
import logging
import sys
import time
from .transcript import TranscriptCollector
from .deepgram_client import DeepgramVoiceProcessor
from .elevenlabs_client import ElevenLabsTextToSpeech

logger = logging.getLogger(__name__)

class VoiceAssistant:
    """Main Voice Assistant handling audio input and AI response"""

    def __init__(self, agent, voice_id=None, enable_audio=True):
        self.agent = agent
        self.transcript_collector = TranscriptCollector()
        self.enable_audio = enable_audio
        self.tts = ElevenLabsTextToSpeech(voice_id) if enable_audio else None
        self.deepgram_processor = DeepgramVoiceProcessor(self.transcript_collector, self.process_voice_input)
        self.response_text = ""

    async def process_voice_input(self, text):
        """Handle AI processing of user command"""
        if not text.strip():
            return

        # Process AI response
        await self._get_agent_response(text)

        # Handle code execution if needed
        await self._handle_code_execution()

    async def _get_agent_response(self, text, new_session=True):
        """Get response from Agent and handle streaming"""
        try:
            logger.debug(f"Processing user input: '{text}'")

            response = await self.agent.ask(text, new_session=new_session, stream=True)
            
            self.response_text = ""

            if self.enable_audio:
                text_chunks = self._create_text_chunk_generator(response)
                asyncio.create_task(self.tts.text_to_speech_stream(text_chunks))

                for chunk in response:
                    if chunk.content:
                        self.response_text += chunk.content
                        await asyncio.sleep(0.05)
            else:
                for chunk in response:
                    if chunk.content:
                        self.response_text += chunk.content
                        await asyncio.sleep(0.05)

        except Exception as e:
            logger.error(f"Error while processing AI response: {str(e)}", exc_info=True)
            self.response_text = f"Error: {str(e)}"

    async def _handle_code_execution(self):
        """Handle execution of code if AI determines this is needed"""
        should_execute = await self.agent.should_ask_user()
        logger.debug(f"Should execute code: {should_execute}")

        if not should_execute:
            return

        max_loops = 5
        loop_count = 0

        while await self.agent.should_ask_user() and not self.agent.should_stop_follow_up(loop_count, max_loops):
            logger.debug(f"Auto-executing code (loop {loop_count+1}/{max_loops})")

            results = self.agent.confirm()
            if not results:
                logger.debug("No results from execution")
                break

            self._display_execution_results(results)

            await self._process_follow_up(results)
            await asyncio.sleep(0)

            loop_count += 1

        if loop_count >= max_loops:
            logger.warning("Reached maximum follow-up iterations")
            self.response_text = "\n\n*Reached maximum number of execution steps*"

    def _display_execution_results(self, results):
        """Display execution results"""
        result_text = "\n\n**Execution Results:**\n```\n" + "\n".join(results) + "\n```"
        self.response_text = result_text

    async def _process_follow_up(self, results):
        """Process follow-up questions based on execution results"""
        logger.debug("Getting follow-up")
        follow_up = await self.agent.get_follow_up(results)

        if follow_up:
            logger.debug(f"Follow-up question: '{follow_up}'")
            await self._get_agent_response(follow_up, new_session=False)

    async def run(self):
        """Run the voice assistant"""
        logger.info("Setting up voice assistant...")
        
        # Setup Deepgram for voice recognition
        try:
            await self.deepgram_processor.setup_deepgram()
            logger.info("Voice recognition ready")
        except Exception as e:
            logger.error(f"Failed to initialize voice recognition: {e}", exc_info=True)
            return
        
        # Print welcome message
        print("\n" + "="*50)
        print("PyAsco Voice Assistant is ready!")
        print("Speak to interact with the assistant.")
        print("Press Ctrl+C to exit.")
        print("="*50 + "\n")
        
        # Keep the assistant running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Voice assistant task cancelled")
        except Exception as e:
            logger.error(f"Error in voice assistant main loop: {e}", exc_info=True)
    
    async def cleanup(self):
        """Clean up resources before shutdown"""
        logger.info("Cleaning up voice assistant resources...")
        
        # Close Deepgram connection if it exists
        if hasattr(self.deepgram_processor, 'close'):
            await self.deepgram_processor.close()
        
        # Close TTS connection if it exists
        if self.tts and hasattr(self.tts, 'close'):
            await self.tts.close()
