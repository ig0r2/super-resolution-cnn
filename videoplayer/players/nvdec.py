import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base_cv2 import _BaseCv2Player
from ..decode.decoder import NvDecoder


class _DecodeReader(threading.Thread):
    """
    Advances an NVDEC decode cursor at real-time pace on a background thread and always exposes
    only the latest decoded frame (an RGB (3,H,W) uint8 CUDA tensor). Mirrors the frame-skipping
    behaviour of the OpenCV player's _FrameReader: if the SR step is slower than the video's
    native framerate, the main loop just picks up the newest frame and older ones are dropped.
    """

    def __init__(self, decoder: NvDecoder):
        super().__init__(daemon=True)
        self.decoder = decoder
        self.frame_time = 1.0 / decoder.fps if decoder.fps > 0 else 1.0 / 30.0

        self.lock = threading.Lock()
        self.frame = None
        self.index = 0
        self.frame_id = 0

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
                self.frame_id += 1

            cursor += 1
            next_time += self.frame_time
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()  # fell behind, don't try to catch up

    def get_latest(self):
        with self.lock:
            return self.frame, self.frame_id, self.index

    def request_seek(self, target_index: int):
        with self.lock:
            self._seek_request = max(0, min(target_index, len(self.decoder) - 1))

    def set_paused(self, paused: bool):
        self.paused = paused

    def stop(self):
        self.running = False


class VideoPlayerNvdecCV2(_BaseCv2Player):
    """
    GPU-decode video player: NVDEC decode -> TensorRT super-resolution -> optional GPU bicubic
    downscale, all on the GPU, with a single device->host copy at the very end for cv2 display.

    upscale_fn takes an RGB (3,H,W) uint8 CUDA tensor and returns an RGB (3,H*s,W*s) uint8 CUDA
    tensor. Same controls as the OpenCV player (play/pause, seek, fullscreen). Frames are decoded
    on a background thread at real-time pace, so a slow SR step skips frames instead of lagging.
    """

    config_desc = "NVDEC decode + TensorRT SR + cv2 display"

    def __init__(self, video_path, target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True, gpu_id: int = 0):
        print(f"[videoplayer] Opening video {Path(video_path).name} (NVDEC) ...")
        self.decoder = NvDecoder(str(video_path), gpu_id=gpu_id)

        fps = self.decoder.fps
        frame_count = len(self.decoder)
        frame_size = (self.decoder.height, self.decoder.width)

        super().__init__(video_path, fps, frame_count, frame_size,
                         target_size=target_size, enable_audio=enable_audio,
                         start_fullscreen=start_fullscreen)

    def _make_reader(self):
        return _DecodeReader(self.decoder)

    def _to_display(self, out_chw: torch.Tensor):
        """(3,H*s,W*s) uint8 RGB CUDA -> BGR (H,W,3) uint8 numpy, downscaled on the GPU if needed."""
        x = out_chw.unsqueeze(0).float()
        if self.target_size is not None:
            x = F.interpolate(x, size=self.target_size, mode="bicubic", align_corners=False)
        x = x.clamp(0.0, 255.0).to(torch.uint8).squeeze(0)  # (3,Ht,Wt)
        bgr = x[[2, 1, 0]].permute(1, 2, 0).contiguous()  # RGB->BGR, CHW->HWC
        return bgr.cpu().numpy()

    def _produce_display(self, frame):
        return self._to_display(self.upscale_fn(frame))
