# elevenlabs_client.py
import os
import json
import logging
import asyncio
import websockets
import base64
import subprocess
import shutil

logger = logging.getLogger(__name__)

class ElevenLabsTextToSpeech:
    """Handles text-to-speech conversion using ElevenLabs API"""

    def __init__(self, voice_id=None):
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.enable_audio = self._is_mpv_installed()
        
    def _is_mpv_installed(self):
        """Check if mpv is installed on the system"""
        if not shutil.which("mpv"):
            logger.warning("mpv not found, audio output disabled. Install mpv to enable audio: https://mpv.io/installation/")
            return False
        return True

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
        if not self.enable_audio:
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

    async def text_to_speech_stream(self, text_iterator):
        """Send text to ElevenLabs and stream the returned audio."""
        if not self.elevenlabs_api_key:
            logger.error("Missing ElevenLabs API key.")
            return
            
        if not self.enable_audio:
            logger.warning("Audio output disabled because mpv is not installed")
            return
        
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?model_id=eleven_flash_v2_5"
       
        try:
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

                # Start streaming audio in background
                listen_task = asyncio.create_task(self._stream_audio(listen()))

                # Send text chunks
                async for text in self._text_chunker(text_iterator):
                    await websocket.send(json.dumps({"text": text}))

                # Signal end of text
                await websocket.send(json.dumps({"text": ""}))
                
                # Wait for audio streaming to complete
                await listen_task
                
        except Exception as e:
            logger.error(f"Error in text-to-speech streaming: {str(e)}")
            
    async def close(self):
        """Clean up resources"""
        pass  # Currently no resources to clean up
