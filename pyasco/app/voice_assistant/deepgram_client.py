# deepgram_client.py

import os
import logging
import asyncio
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions, Microphone

logger = logging.getLogger(__name__)

class DeepgramVoiceProcessor:
    """Handles real-time voice transcription using Deepgram"""
    
    def __init__(self, transcript_collector, process_voice_callback):
        self.transcript_collector = transcript_collector
        self.process_voice_callback = process_voice_callback
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "")
        self.dg_connection = None
        self.microphone = None

    async def setup_deepgram(self):
        """Setup Deepgram connection"""
        if not self.deepgram_api_key:
            logger.error("Deepgram API key not set.")
            raise ValueError("Deepgram API key is required. Set DEEPGRAM_API_KEY in your environment.")

        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(self.deepgram_api_key, config)
        self.dg_connection = deepgram.listen.asynclive.v("1")

        async def on_message(this, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript

            if not sentence.strip():
                return

            if not result.speech_final:
                self.transcript_collector.update_partial(sentence)
            else:
                self.transcript_collector.add_part(sentence)
                full_sentence = self.transcript_collector.get_full_transcript()

                self.transcript_collector.add_sentence(full_sentence)
                self.transcript_collector.update_partial("")
                
                await self.process_voice_callback(self.transcript_collector.get_combined_input())

                self.transcript_collector.reset()

        async def on_error(this, error, **kwargs):
            logger.error(f"Deepgram error: {error}")

        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=True
        )

        await self.dg_connection.start(options)
        
    def start_microphone(self):
        """Start the microphone to capture audio input"""
        if not self.dg_connection:
            logger.error("Deepgram connection not initialized")
            raise RuntimeError("Must call setup_deepgram() before starting microphone")
            
        logger.debug("Opening microphone stream")
        self.microphone = Microphone(self.dg_connection.send)
        self.microphone.start()
        logger.info("Microphone activated and listening")
        
    def is_microphone_active(self):
        """Check if the microphone is active"""
        return self.microphone and self.microphone.is_active()
        
    async def close(self):
        """Clean up resources"""
        logger.debug("Cleaning up Deepgram resources")
        if self.microphone:
            self.microphone.finish()
        if self.dg_connection:
            self.dg_connection.finish()
