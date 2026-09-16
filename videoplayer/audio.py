import subprocess
import threading
import time
from pathlib import Path
from typing import Optional


def _log(msg: str):
    print(f"[videoplayer] {msg}")


class AudioTrack:
    """
    Extracts a video's audio track to a cached WAV via ffmpeg (once) and plays it back through
    sounddevice, kept in sync with VideoPlayer's pause/seek state. Fails soft: if ffmpeg,
    sounddevice, soundfile, or the audio track itself aren't available, `available` stays False
    and the player just runs silently.
    """

    def __init__(self, video_path: Path, ffmpeg_path: str = "ffmpeg"):
        self.available = False
        self.data = None
        self.samplerate = 0
        self.channels = 0

        self._lock = threading.Lock()
        self._pos = 0
        self._paused = True
        self._stream = None
        self._sd = None

        cache_dir = Path(__file__).resolve().parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{video_path.stem}_audio.wav"

        if not cache_path.exists():
            _log(f"Extracting audio track via ffmpeg -> {cache_path.name} ...")
            t0 = time.perf_counter()
            try:
                subprocess.run(
                    [ffmpeg_path, "-y", "-i", str(video_path), "-vn",
                     "-acodec", "pcm_s16le", str(cache_path)],
                    check=True, capture_output=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
                _log(f"Audio extraction failed ({e}), playing without sound.")
                return
            _log(f"Audio extracted in {time.perf_counter() - t0:.1f}s")
        else:
            _log(f"Reusing cached audio track {cache_path.name}")

        try:
            import soundfile as sf
            data, samplerate = sf.read(str(cache_path), dtype="float32", always_2d=True)
        except Exception as e:
            _log(f"Could not load audio track ({e}), playing without sound.")
            return

        try:
            import sounddevice as sd
        except Exception as e:
            _log(f"sounddevice not available ({e}), playing without sound.")
            return

        self.data = data
        self.samplerate = samplerate
        self.channels = data.shape[1]
        self._sd = sd
        self.available = True
        _log(f"Audio track loaded ({self.channels}ch @ {self.samplerate}Hz).")

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            if self._paused or self.data is None:
                outdata.fill(0)
                return
            start = self._pos
            end = start + frames
            chunk = self.data[start:end]
            n = len(chunk)
            outdata[:n] = chunk
            if n < frames:
                outdata[n:] = 0
            self._pos = end

    def start(self):
        if not self.available:
            return
        self._stream = self._sd.OutputStream(
            samplerate=self.samplerate, channels=self.channels, dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def set_paused(self, paused: bool):
        if not self.available:
            return
        with self._lock:
            self._paused = paused

    def seek(self, seconds: float):
        if not self.available:
            return
        with self._lock:
            self._pos = max(0, min(int(seconds * self.samplerate), len(self.data)))

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
