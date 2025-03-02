# elevenlabs_client.py
import os
import json
import logging
import asyncio
import websockets
import base64
import subprocess

logger = logging.getLogger(__name__)

class ElevenLabsTextToSpeech:
    """Handles text-to-speech conversion using ElevenLabs API"""

    def __init__(self, voice_id=None):
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

    async def text_to_speech_stream(self, text_iterator):
        """Send text to ElevenLabs and stream the returned audio."""
        if not self.elevenlabs_api_key:
            logger.error("Missing ElevenLabs API key.")
            return
        
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id=eleven_flash_v2_5"
       
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": self.elevenlabs_api_key,
            }))
            
            async for text in text_iterator:
                await websocket.send(json.dumps({"text": text}))

            await websocket.send(json.dumps({"text": ""}))
