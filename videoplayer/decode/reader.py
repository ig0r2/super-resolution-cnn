import threading
import time
from typing import Optional


class _DecodeReader(threading.Thread):
    """
    Advances a decode cursor at real-time pace on a background thread and always exposes only the
    latest decoded frame, so if the SR step is slower than the video's native framerate the main
    loop just picks up the newest frame and older ones are dropped (frames are skipped instead of
    playback slowing down).

    Works with any decoder with the `len()` / `fps` / `frame(index)` interface: NvDecoder (frame =
    RGB (3,H,W) uint8 CUDA tensor), PyAVDecoder / CV2Decoder ((H,W,3) BGR numpy).
    """

    def __init__(self, decoder):
        super().__init__(daemon=True)
        self.decoder = decoder
        self.frame_time = 1.0 / decoder.fps if decoder.fps > 0 else 1.0 / 30.0

        self.lock = threading.Lock()
        self.frame = None
        self.index = 0
        self.frame_counter = 0

        self.running = True
        self.paused = False
        self._seek_request: Optional[int] = None

    def run(self):
        next_time = time.perf_counter()
        cursor = 0

        while self.running:
            if self.paused:
                time.sleep(0.01)
                next_time = time.perf_counter()
                continue

            with self.lock:
                seek_to = self._seek_request
                self._seek_request = None
            if seek_to is not None:
                cursor = seek_to
                next_time = time.perf_counter()

            if cursor >= len(self.decoder):
                self.running = False
                break

            frame = self.decoder.frame(cursor)

            with self.lock:
                self.frame = frame
                self.index = cursor
                self.frame_counter += 1

            cursor += 1
            next_time += self.frame_time
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()  # fell behind, don't try to catch up

    def get_latest(self):
        with self.lock:
            return self.frame, self.frame_counter, self.index

    def request_seek(self, target_index: int):
        with self.lock:
            self._seek_request = max(0, min(target_index, len(self.decoder) - 1))

    def set_paused(self, paused: bool):
        self.paused = paused

    def stop(self):
        self.running = False
