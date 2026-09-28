from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base_cv2 import _BaseCv2Player


class VideoPlayerNvdecCV2(_BaseCv2Player):
    """
    GPU-decode video player: NVDEC decode -> TensorRT super-resolution -> optional GPU bicubic
    downscale, all on the GPU, with a single device->host copy at the very end for cv2 display.

    upscale_fn (a TRTBackendNVDEC) takes an RGB (3,H,W) uint8 CUDA tensor and returns a (1,3,H*s,W*s) fp16 RGB frame, which the GPU bicubic runs on directly when it is
    downscaled for the screen, and which is converted to uint8 BGR on the GPU before the copy to
    host. Same controls as the PyAV player (play/pause, seek, fullscreen). Frames are decoded on a background
    thread at real-time pace, so a slow SR step skips frames instead of lagging.
    """

    config_desc = "NVDEC decode + TensorRT SR + cv2 display"

    def __init__(self, video_path, target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True):
        from ..decode.decoder_nvdec import NvDecoder

        print(f"[videoplayer] Opening video {Path(video_path).name} (NVDEC) ...")
        super().__init__(video_path, NvDecoder(str(video_path)), target_size=target_size,
                         enable_audio=enable_audio, start_fullscreen=start_fullscreen)

    def _produce_display(self, frame):
        """NVDEC frame -> SR -> (1,3,H*s,W*s) fp16 RGB [0,255] CUDA -> (H,W,3) uint8 BGR numpy,
        bicubic-downscaled to target_size on the GPU first if one is set."""
        out = self.upscale_fn(frame)
        if self.target_size is not None:
            out = F.interpolate(out, size=self.target_size, mode="bicubic", align_corners=False)
            out.clamp_(0.0, 255.0)
        out = out.to(torch.uint8).squeeze(0)[[2, 1, 0]].permute(1, 2, 0).contiguous()
        return out.cpu().numpy()
