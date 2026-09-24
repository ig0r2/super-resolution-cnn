"""
CUDA<->OpenGL variant of the NVDEC player: the SR output is shown straight from GPU memory via a
CUDA-registered GL texture (see gl_display.GLDisplay), with no device->host copy and no cv2.

Reuses VideoPlayerNvdecCV2 for decode/reader/audio/seek and _BaseGlPlayer for the glfw display loop;
only the surface hooks are player-specific. Frame stats (FPS, position) go to the window title bar
instead of being drawn onto the frame, which avoids a CPU text-draw pass and any GL text rendering.
"""

import glfw
import torch
import torch.nn.functional as F

from .base_gl import _BaseGlPlayer
from ..backends.gl_display import GLDisplay
from .nvdec import VideoPlayerNvdecCV2


class VideoPlayerNvdecGL(_BaseGlPlayer, VideoPlayerNvdecCV2):
    config_desc = "NVDEC decode + TensorRT SR + CUDA-GL zero-copy display"

    def _downscale(self, out_chw: torch.Tensor) -> torch.Tensor:
        """(3,H*s,W*s) uint8 RGB CUDA -> (3,Ht,Wt) uint8 RGB CUDA, GPU bicubic if a target is set."""
        if self.target_size is None:
            return out_chw
        x = out_chw.unsqueeze(0).float()
        x = F.interpolate(x, size=self.target_size, mode="bicubic", align_corners=False)
        return x.clamp(0.0, 255.0).to(torch.uint8).squeeze(0)

    def _gl_open(self, on_key):
        # Window starts at the display target if we have one, else the native frame size.
        win_h, win_w = self.target_size if self.target_size is not None else self.frame_size
        display = GLDisplay(win_w, win_h, self.window_name, fullscreen=self.fullscreen)
        glfw.set_key_callback(display.window, on_key)
        return display

    def _gl_infer(self, frame):
        self._surface.upload(self._downscale(self.upscale_fn(frame)))

    def _gl_present(self, title: str, have_frame: bool):
        if have_frame:
            self._surface.render()
        self._surface.set_title(title)
