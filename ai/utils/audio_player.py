"""
Realtime audio player cho TTS streaming
Hỗ trợ cả PCM và MP3 format
"""

import io
import numpy as np
import sounddevice as sd
from typing import Optional, Literal


class RealtimeAudioPlayer:
    """Phát audio realtime từ PCM hoặc MP3 bytes stream"""

    def __init__(
        self,
        samplerate: int = 24000,
        channels: int = 1,
        audio_format: Literal["pcm", "mp3"] = "pcm",
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.audio_format = audio_format
        self.stream: Optional[sd.OutputStream] = None
        self.leftover = b""  # Buffer cho bytes lẻ (PCM) hoặc incomplete MP3 frames
        self._mp3_buffer = b""  # Buffer tích lũy MP3 để decode

    def start(self):
        """Bắt đầu stream audio"""
        if self.stream is None:
            self.stream = sd.OutputStream(
                samplerate=self.samplerate, channels=self.channels, dtype="int16"
            )
            self.stream.start()
        self.leftover = b""
        self._mp3_buffer = b""

    def _decode_mp3_to_pcm(self, mp3_bytes: bytes) -> Optional[np.ndarray]:
        """Decode MP3 bytes sang PCM numpy array"""
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            # Resample nếu cần
            if audio.frame_rate != self.samplerate:
                audio = audio.set_frame_rate(self.samplerate)
            if audio.channels != self.channels:
                audio = audio.set_channels(self.channels)
            # Convert sang numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
            return samples
        except Exception:
            return None

    def play_chunk(self, chunk: bytes):
        """Phát một chunk audio"""
        if self.stream is None:
            self.start()

        if self.audio_format == "mp3":
            self._play_mp3_chunk(chunk)
        else:
            self._play_pcm_chunk(chunk)

    def _play_pcm_chunk(self, chunk: bytes):
        """Phát PCM chunk trực tiếp"""
        audio_bytes = self.leftover + chunk

        # int16 cần 2 bytes, giữ lại phần lẻ
        remainder = len(audio_bytes) % 2
        if remainder:
            self.leftover = audio_bytes[-remainder:]
            audio_bytes = audio_bytes[:-remainder]
        else:
            self.leftover = b""

        if audio_bytes:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            self.stream.write(audio_data)

    def _play_mp3_chunk(self, chunk: bytes):
        """Tích lũy MP3 chunks và decode khi đủ data"""
        self._mp3_buffer += chunk

        # Thử decode sau mỗi vài KB (MP3 cần đủ frames)
        # MP3 frame ~400-1000 bytes, buffer ~4KB trước khi decode
        if len(self._mp3_buffer) >= 4096:
            pcm_data = self._decode_mp3_to_pcm(self._mp3_buffer)
            if pcm_data is not None and len(pcm_data) > 0:
                self.stream.write(pcm_data)
                self._mp3_buffer = b""  # Reset buffer sau khi decode thành công

    def stop(self):
        """Dừng và đóng stream"""
        if self.stream:
            # Phát nốt buffer còn lại
            if self.audio_format == "mp3" and self._mp3_buffer:
                pcm_data = self._decode_mp3_to_pcm(self._mp3_buffer)
                if pcm_data is not None and len(pcm_data) > 0:
                    self.stream.write(pcm_data)
                self._mp3_buffer = b""
            elif self.leftover:
                audio_data = np.frombuffer(self.leftover + b"\x00", dtype=np.int16)
                self.stream.write(audio_data)
                self.leftover = b""

            import time

            time.sleep(0.5)  # Chờ buffer flush

            self.stream.stop()
            self.stream.close()
            self.stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# Singleton player cho tiện dùng
_player: Optional[RealtimeAudioPlayer] = None


def get_audio_player() -> RealtimeAudioPlayer:
    """Lấy singleton audio player"""
    global _player
    if _player is None:
        _player = RealtimeAudioPlayer()
    return _player


def play_audio_chunk(chunk: bytes):
    """Shortcut để phát một chunk audio"""
    player = get_audio_player()
    player.play_chunk(chunk)


def stop_audio_player():
    """Dừng audio player"""
    global _player
    if _player:
        _player.stop()
        _player = None
