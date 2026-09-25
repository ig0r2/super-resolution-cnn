import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2

from .base_cv2 import _BaseCv2Player


class _FrameReader(threading.Thread):
    """
    Reads frames from a cv2.VideoCapture at real-time pace on a background thread and always
    exposes only the latest one. This is what lets the main loop skip frames
    instead of slowing down when upscale_fn is slower than the video's native framerate.
    """

    def __init__(self, cap: cv2.VideoCapture, fps: float):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0

        self.lock = threading.Lock()
        self.frame = None
        self.pos_frames = 0.0
        self.frame_counter = 0

        self.running = True
        self.paused = False
        self._seek_request: Optional[float] = None

    def run(self):
        next_time = time.perf_counter()

        while self.running:
            if self.paused:
                time.sleep(0.01)
                next_time = time.perf_counter()
                continue

            with self.lock:
                seek_to = self._seek_request
                self._seek_request = None
            if seek_to is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                next_time = time.perf_counter()

            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            with self.lock:
                self.frame = frame
                self.pos_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.frame_counter += 1

            next_time += self.frame_time
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()  # fell behind, don't try to catch up

    def get_latest(self):
        with self.lock:
            return self.frame, self.frame_counter, self.pos_frames

    def request_seek(self, target_frame: float):
        with self.lock:
            self._seek_request = target_frame

    def set_paused(self, paused: bool):
        self.paused = paused

    def stop(self):
        self.running = False


class VideoPlayerCV2(_BaseCv2Player):
    """
    OpenCV based video player with play/pause, seeking, and a fullscreen toggle.

    upscale_fn is applied to the latest available native-resolution BGR frame; if target_size
    is given, the upscaled frame is bicubic-downscaled to that exact (h, w) so it fits the
    screen. Frames are read on a background thread at real-time pace, so if upscale_fn is
    slower than the video's native framerate, frames are skipped rather than played back slow.
    """

    config_desc = "OpenCV (cv2) decode + cv2 display"

    def __init__(self, video_path, target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True):
        print(f"[videoplayer] Opening video {Path(video_path).name} ...")
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

        super().__init__(video_path, fps, frame_count, frame_size,
                         target_size=target_size, enable_audio=enable_audio, start_fullscreen=start_fullscreen)

    def _make_reader(self):
        return _FrameReader(self.cap, self.fps)

    def _produce_display(self, frame):
        output = self.upscale_fn(frame)
        if self.target_size is not None:
            th, tw = self.target_size
            output = cv2.resize(output, (tw, th), interpolation=cv2.INTER_CUBIC)
        return output

    def _release_source(self):
        self.cap.release()
