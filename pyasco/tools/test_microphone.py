"""
Test script to verify microphone functionality
"""

import pyaudio
import wave
import os
import time
from rich.console import Console

console = Console()

def test_microphone():
    """Test if the microphone is working by recording a short audio clip"""
    
    console.print("[bold green]Testing microphone...[/bold green]")
    
    # Audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "test_recording.wav"
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    try:
        # Get information about available audio devices
        info = audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        console.print(f"Found {num_devices} audio devices:")
        
        # List all audio input devices
        input_devices = []
        for i in range(num_devices):
            device_info = audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                input_devices.append((i, device_info.get('name')))
                console.print(f"  [{i}] {device_info.get('name')}")
        
        if not input_devices:
            console.print("[bold red]No input devices found![/bold red]")
            return False
        
        # Open stream
        console.print("\n[bold yellow]Recording for 3 seconds...[/bold yellow]")
        stream = audio.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           input=True,
                           frames_per_buffer=CHUNK)
        
        # Record audio
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            if i % 10 == 0:
                console.print(".", end="")
        
        console.print("\n[bold green]Recording complete![/bold green]")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        
        # Save the recorded audio to a WAV file
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        console.print(f"[bold green]Audio saved to {WAVE_OUTPUT_FILENAME}[/bold green]")
        
        # Check if the file has content
        file_size = os.path.getsize(WAVE_OUTPUT_FILENAME)
        if file_size > 1000:  # Arbitrary threshold to check if recording has content
            console.print(f"[bold green]Microphone test successful! File size: {file_size} bytes[/bold green]")
            return True
        else:
            console.print(f"[bold yellow]Warning: Recording file is very small ({file_size} bytes). Microphone might not be capturing audio.[/bold yellow]")
            return False
            
    except Exception as e:
        console.print(f"[bold red]Error testing microphone: {str(e)}[/bold red]")
        return False
    finally:
        # Terminate PyAudio
        audio.terminate()

if __name__ == "__main__":
    test_microphone()
