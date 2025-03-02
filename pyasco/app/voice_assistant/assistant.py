import asyncio
import logging
import sys
import time
import traceback
from .transcript import TranscriptCollector
from .deepgram_client import DeepgramVoiceProcessor
from .elevenlabs_client import ElevenLabsTextToSpeech

logger = logging.getLogger(__name__)

class VoiceAssistant:
    """Main Voice Assistant handling audio input and AI response"""

    def __init__(self, agent, voice_id=None, enable_audio=True):
        logger.info("Initializing VoiceAssistant")
        self.agent = agent
        self.transcript_collector = TranscriptCollector()
        self.enable_audio = enable_audio
        logger.info(f"Audio output enabled: {enable_audio}")
        
        if enable_audio:
            logger.info(f"Initializing text-to-speech with voice ID: {voice_id}")
            self.tts = ElevenLabsTextToSpeech(voice_id)
        else:
            logger.info("Text-to-speech disabled")
            self.tts = None
            
        logger.info("Initializing Deepgram voice processor")
        self.deepgram_processor = DeepgramVoiceProcessor(self.transcript_collector, self.process_voice_input)
        self.response_text = ""
        logger.info("VoiceAssistant initialization complete")

    async def process_voice_input(self, text):
        """Handle AI processing of user command"""
        logger.info(f"Processing voice input: '{text}'")
        
        if not text.strip():
            logger.debug("Empty input received, ignoring")
            return

        # Process AI response
        logger.info("Getting agent response")
        await self._get_agent_response(text)

        # Handle code execution if needed
        logger.info("Checking for code execution")
        await self._handle_code_execution()
        
        logger.info("Voice input processing complete")

    async def _get_agent_response(self, text, new_session=True):
        """Get response from Agent and handle streaming"""
        try:
            logger.info(f"Getting agent response for: '{text}' (new_session={new_session})")

            # Get streaming response from agent
            logger.debug("Calling agent.ask() with streaming enabled")
            response = await self.agent.ask(text, new_session=new_session, stream=True)
            
            self.response_text = ""
            logger.info("Processing agent response stream")

            # Create a copy of the response generator for TTS
            if self.enable_audio and self.tts:
                logger.info("Audio enabled - collecting response for TTS")
                # We need to create two separate iterators from the same response
                # One for TTS and one for display
                response_list = []
                chunk_count = 0
                
                # Collect all response chunks
                logger.debug("Collecting response chunks")
                for chunk in response:
                    if chunk.content:
                        chunk_count += 1
                        response_list.append(chunk.content)
                        self.response_text += chunk.content
                        await asyncio.sleep(0.05)
                
                logger.info(f"Collected {chunk_count} response chunks, total length: {len(self.response_text)} chars")
                
                # Create a generator for TTS from the collected chunks
                async def text_chunks():
                    logger.debug("Starting text chunks generator for TTS")
                    for chunk in response_list:
                        yield chunk
                
                # Start TTS in background
                logger.info("Starting text-to-speech stream in background")
                asyncio.create_task(self.tts.text_to_speech_stream(text_chunks()))
            else:
                # Just display the response without TTS
                logger.info("Audio disabled - processing response for display only")
                chunk_count = 0
                for chunk in response:
                    if chunk.content:
                        chunk_count += 1
                        self.response_text += chunk.content
                        await asyncio.sleep(0.05)
                
                logger.info(f"Processed {chunk_count} response chunks, total length: {len(self.response_text)} chars")

            logger.info("Agent response processing complete")

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error while processing AI response: {str(e)}\n{error_details}")
            self.response_text = f"Error: {str(e)}"

    async def _handle_code_execution(self):
        """Handle execution of code if AI determines this is needed"""
        logger.info("Checking if code execution is needed")
        should_execute = await self.agent.should_ask_user()
        logger.info(f"Should execute code: {should_execute}")

        if not should_execute:
            logger.info("No code execution needed")
            return

        max_loops = 5
        loop_count = 0
        logger.info(f"Starting code execution loop (max loops: {max_loops})")

        while await self.agent.should_ask_user() and not self.agent.should_stop_follow_up(loop_count, max_loops):
            logger.info(f"Auto-executing code (loop {loop_count+1}/{max_loops})")

            logger.debug("Calling agent.confirm() to execute code")
            results = self.agent.confirm()
            
            if not results:
                logger.warning("No results from execution")
                break

            logger.info(f"Received execution results: {len(results)} items")
            logger.debug(f"Execution results: {results}")
            self._display_execution_results(results)

            logger.info("Processing follow-up based on execution results")
            await self._process_follow_up(results)
            await asyncio.sleep(0)

            loop_count += 1
            logger.debug(f"Completed execution loop {loop_count}/{max_loops}")

        if loop_count >= max_loops:
            logger.warning("Reached maximum follow-up iterations")
            self.response_text = "\n\n*Reached maximum number of execution steps*"
            
        logger.info(f"Code execution complete after {loop_count} loops")

    def _display_execution_results(self, results):
        """Display execution results"""
        logger.info(f"Displaying execution results ({len(results)} items)")
        result_text = "\n\n**Execution Results:**\n```\n" + "\n".join(results) + "\n```"
        self.response_text = result_text
        logger.debug(f"Result text: {result_text}")

    async def _process_follow_up(self, results):
        """Process follow-up questions based on execution results"""
        logger.info("Getting follow-up from agent")
        follow_up = await self.agent.get_follow_up(results)

        if follow_up:
            logger.info(f"Received follow-up question: '{follow_up}'")
            logger.info("Processing follow-up response (continuing session)")
            await self._get_agent_response(follow_up, new_session=False)
        else:
            logger.info("No follow-up question received")

    async def run(self):
        """Run the voice assistant"""
        logger.info("Starting voice assistant...")
        
        # Setup Deepgram for voice recognition
        try:
            logger.info("Setting up Deepgram voice recognition")
            await self.deepgram_processor.setup_deepgram()
            
            logger.info("Starting microphone capture")
            self.deepgram_processor.start_microphone()
            logger.info("Voice recognition ready and listening")
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Failed to initialize voice recognition: {e}\n{error_details}")
            return
        
        # Print welcome message
        welcome_message = "\n" + "="*50 + "\nPyAsco Voice Assistant is ready!\nSpeak to interact with the assistant.\nPress Ctrl+C to exit.\n" + "="*50 + "\n"
        print(welcome_message)
        logger.info("Voice assistant ready and running")
        
        # Keep the assistant running until interrupted
        try:
            logger.info("Entering main loop")
            loop_count = 0
            while True:
                await asyncio.sleep(1)
                loop_count += 1
                
                # Log status every minute
                if loop_count % 60 == 0:
                    logger.debug(f"Voice assistant running for {loop_count} seconds")
                
                # Check if microphone is still active
                if not self.deepgram_processor.is_microphone_active():
                    logger.warning("Microphone is no longer active, shutting down")
                    break
        except asyncio.CancelledError:
            logger.info("Voice assistant task cancelled")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down")
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in voice assistant main loop: {e}\n{error_details}")
        
        logger.info("Voice assistant main loop exited")
    
    async def cleanup(self):
        """Clean up resources before shutdown"""
        logger.info("Cleaning up voice assistant resources...")
        
        try:
            # Close Deepgram connection if it exists
            logger.info("Closing Deepgram connection")
            await self.deepgram_processor.close()
            
            # Close TTS connection if it exists
            if self.tts and hasattr(self.tts, 'close'):
                logger.info("Closing text-to-speech connection")
                await self.tts.close()
                
            # Clean up agent resources
            logger.info("Cleaning up agent resources")
            self.agent.cleanup()
            
            logger.info("All resources cleaned up successfully")
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error during cleanup: {e}\n{error_details}")
