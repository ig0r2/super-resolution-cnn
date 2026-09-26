from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base_cv2 import _BaseCv2Player


class VideoPlayerNvdecCV2(_BaseCv2Player):
    """
    GPU-decode video player: NVDEC decode -> TensorRT super-resolution -> optional GPU bicubic
    downscale, all on the GPU, with a single device->host copy at the very end for cv2 display.

    upscale_fn takes an RGB (3,H,W) uint8 CUDA tensor; build it with ``output=sr_output()`` (after
    configure_scale): a BGR (H*s,W*s,3) uint8 frame shown as is, or, when it is downscaled for
    the screen, a (1,3,H*s,W*s) fp16 RGB frame the GPU bicubic runs on directly. Same controls as the PyAV player (play/pause, seek, fullscreen). Frames are decoded on a background
    thread at real-time pace, so a slow SR step skips frames instead of lagging.
    """

    config_desc = "NVDEC decode + TensorRT SR + cv2 display"

    def __init__(self, video_path, target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True):
        from ..decode.nvdec_decoder import NvDecoder

        print(f"[videoplayer] Opening video {Path(video_path).name} (NVDEC) ...")
        super().__init__(video_path, NvDecoder(str(video_path)), target_size=target_size,
                         enable_audio=enable_audio, start_fullscreen=start_fullscreen)

    def sr_output(self):
        """Engine output layout for TRTBackendNVDEC (call after configure_scale)."""
        return "rgb_f16" if self.target_size is not None else "bgr"

    def _to_display(self, out: torch.Tensor):
        """SR output CUDA -> (H,W,3) uint8 BGR numpy: (H*s,W*s,3) uint8 BGR is copied as is,
        (1,3,H*s,W*s) fp16 RGB is bicubic-downscaled to target_size on the GPU first."""
        if self.target_size is not None:
            x = F.interpolate(out, size=self.target_size, mode="bicubic", align_corners=False)
            out = x.clamp(0.0, 255.0).to(torch.uint8).squeeze(0)[[2, 1, 0]].permute(1, 2, 0).contiguous()
        return out.cpu().numpy()

    def _produce_display(self, frame):
        return self._to_display(self.upscale_fn(frame))
