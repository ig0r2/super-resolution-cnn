from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2

from .base_cv2 import _BaseCv2Player
from ..backends.cv2_backends import Runtype, make_backend

Decoder = Literal["pyav", "cv2"]


def open_cpu_decoder(video_path, decoder: Decoder):
    """PyAVDecoder or CV2Decoder (both give (H,W,3) BGR frames)"""
    if decoder == "pyav":
        from ..decode.decoder_pyav import PyAVDecoder
        return PyAVDecoder(str(video_path))
    if decoder == "cv2":
        from ..decode.decoder_cv2 import CV2Decoder
        return CV2Decoder(str(video_path))
    raise ValueError(f"Unknown decoder {decoder!r} (expected 'pyav' or 'cv2')")


class VideoPlayerCV2(_BaseCv2Player):
    """
    CPU decode -> SR backend -> optional cv2 bicubic downscale -> cv2 display, with play/pause,
    seeking and a fullscreen toggle.

    Takes the model (checkpoint name or .pth path) and the runtype directly: the scale is picked for
    the screen and the SR backend is built (make_backend) on construction, so a failing export /
    engine build raises here, before any window opens.

    decoder picks the CPU decoder: "pyav" (FFmpeg via PyAV) or "cv2" (cv2.VideoCapture); both give
    (H,W,3) BGR frames, so either works with the same bgr -> bgr SR backend.
    upscale_fn returns a BGR uint8 (H*s,W*s,3) frame; if target_size is given it is
    bicubic-downscaled to that exact (h, w) to fit the screen. Frames are decoded on a background
    thread at real-time pace, so a slow SR step skips frames instead of slowing playback.
    """

    def __init__(self, video_path, model, runtype: Runtype = "tensorrt", decoder: Decoder = "pyav",
                 candidate_scales=(2, 3, 4), target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True):
        print(f"[videoplayer] Opening video {Path(video_path).name} ({decoder}) ...")
        self.config_desc = f"{decoder} decode + {runtype} SR + cv2 display"
        super().__init__(video_path, open_cpu_decoder(video_path, decoder),
                         target_size=target_size, enable_audio=enable_audio, start_fullscreen=start_fullscreen)

        self.scale = self.configure_scale(candidate_scales)
        self.set_upscale_fn(make_backend(runtype, model, self.size, self.scale))

    def _produce_display(self, frame):
        output = self.upscale_fn(frame)
        if self.target_size is not None:
            th, tw = self.target_size
            output = cv2.resize(output, (tw, th), interpolation=cv2.INTER_CUBIC)
        return output
