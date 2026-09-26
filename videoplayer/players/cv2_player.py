from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2

from .base_cv2 import _BaseCv2Player

Decoder = Literal["pyav", "cv2"]


def open_cpu_decoder(video_path, decoder: Decoder):
    """PyAVDecoder or CV2Decoder (both give (H,W,3) BGR frames); imported lazily so
    `import videoplayer` doesn't pull PyAV in."""
    if decoder == "pyav":
        from ..decode.pyav_decoder import PyAVDecoder
        return PyAVDecoder(str(video_path))
    if decoder == "cv2":
        from ..decode.cv2_decoder import CV2Decoder
        return CV2Decoder(str(video_path))
    raise ValueError(f"Unknown decoder {decoder!r} (expected 'pyav' or 'cv2')")


class VideoPlayerCV2(_BaseCv2Player):
    """
    CPU decode -> SR backend -> optional cv2 bicubic downscale -> cv2 display, with play/pause,
    seeking and a fullscreen toggle.

    decoder picks the CPU decoder: "pyav" (FFmpeg via PyAV) or "cv2" (cv2.VideoCapture); both give
    (H,W,3) BGR frames, so the SR backend is built with `video_io()` (bgr -> bgr) either way.
    upscale_fn returns a BGR uint8 (H*s,W*s,3) frame; if target_size is given it is
    bicubic-downscaled to that exact (h, w) to fit the screen. Frames are decoded on a background
    thread at real-time pace, so a slow SR step skips frames instead of slowing playback.
    """

    def __init__(self, video_path, decoder: Decoder = "pyav", target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True):
        print(f"[videoplayer] Opening video {Path(video_path).name} ({decoder}) ...")
        self.config_desc = f"{decoder} decode + SR + cv2 display"
        super().__init__(video_path, open_cpu_decoder(video_path, decoder),
                         target_size=target_size, enable_audio=enable_audio, start_fullscreen=start_fullscreen)

    @staticmethod
    def video_io():
        """The VideoIO the SR backend must be built with: BGR frames in, BGR out."""
        from ..backends.wrappers import VideoIO
        return VideoIO("bgr", "bgr")

    def _produce_display(self, frame):
        output = self.upscale_fn(frame)
        if self.target_size is not None:
            th, tw = self.target_size
            output = cv2.resize(output, (tw, th), interpolation=cv2.INTER_CUBIC)
        return output
