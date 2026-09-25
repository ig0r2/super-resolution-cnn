import threading
import time
from pathlib import Path


def _log(msg: str):
    print(f"[videoplayer] {msg}")


class AudioTrack:
    """
    Decodes a video's first audio stream in-process with PyAV into memory as packed float32
    mono stays mono, everything else is downmixed to stereo -- and plays it back through sounddevice
    kept in sync with the player's pause/seek state.
    """

    def __init__(self, video_path: Path):
        self.available = False
        self.data = None
        self.samplerate = 0
        self.channels = 0

        self._lock = threading.Lock()
        self._pos = 0
        self._paused = True
        self._stream = None
        self._sd = None

        loaded = self._load(Path(video_path))
        if loaded is None:
            return
        data, samplerate = loaded

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

    @staticmethod
    def _load(video_path: Path):
        """Return ``(float32 (N, channels) samples, samplerate)``, or None to run silent."""
        try:
            import av
            import numpy as np
        except Exception as e:
            _log(f"PyAV not available ({e}), playing without sound.")
            return None

        t0 = time.perf_counter()
        try:
            with av.open(str(video_path)) as container:
                if not container.streams.audio:
                    _log("No audio stream in the video, playing without sound.")
                    return None
                stream = container.streams.audio[0]
                stream.thread_type = "AUTO"
                channels = 1 if stream.channels == 1 else 2
                samplerate = stream.rate
                resampler = av.AudioResampler(format="flt", layout="mono" if channels == 1 else "stereo",
                                              rate=samplerate)

                chunks = []
                for frame in container.decode(stream):
                    for out in resampler.resample(frame):
                        chunks.append(out.to_ndarray())
                for out in resampler.resample(None):  # flush
                    chunks.append(out.to_ndarray())
        except Exception as e:
            _log(f"Audio decode failed ({e}), playing without sound.")
            return None

        if not chunks:
            _log("Audio stream decoded to no samples, playing without sound.")
            return None
        # Packed float32 frames come out as (1, samples * channels), interleaved.
        data = np.concatenate(chunks, axis=1).reshape(-1, channels)
        _log(f"Audio decoded via PyAV in {time.perf_counter() - t0:.1f}s")
        return np.ascontiguousarray(data, dtype=np.float32), samplerate

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
