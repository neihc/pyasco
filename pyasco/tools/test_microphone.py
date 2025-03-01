import wave
from typing import List, Optional, TypeVar, Union, IO

import pyaudio #  type: ignore

WaveWrite = TypeVar("WaveWrite", bound=wave.Wave_write)


def init_recording(
    file_name: Union[str, IO[bytes]] = "sound.wav", mode: Optional[str] = "wb"
) -> WaveWrite:
    wave_file = wave.open(file_name, mode)
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)
    wave_file.setframerate(44100)
    return wave_file


def record(wave_file: WaveWrite, duration: Optional[int] = 3) -> None:
    py_audio = pyaudio.PyAudio()
    audio_stream = py_audio.open(
        rate=44100,  # frames per second,
        channels=1,  # stereo, change to 1 if you want mono
        format=8,  # sample format, 8 bytes. see inspect
        input=True,  # input device flag
        frames_per_buffer=1024,  # 1024 samples per frame
    )
    frames = []
    for _ in range(int(44100 / 1024 * 3)):
        data = audio_stream.read(1024)
        frames.append(data)
    wave_file.writeframes(b"".join(frames))
    audio_stream.close()


if __name__ == "__main__":
    wave_file = init_recording() #  type: ignore
    record(wave_file)
    wave_file.close()
