"""
CUDA<->OpenGL variant of the NVDEC player: the SR output is shown straight from GPU memory via a
CUDA-registered GL texture (see gl_display.GLDisplay), with no device->host copy and no cv2.

_BaseGlPlayer provides NVDEC decode / reader / audio / seek and the glfw display loop; only the
surface hooks are player-specific. Frame stats (FPS, position) go to the window title bar
instead of being drawn onto the frame, which avoids a CPU text-draw pass and any GL text rendering.
"""

import glfw
import torch
import torch.nn.functional as F

from .base_gl import _BaseGlPlayer
from ..backends.gl_display import GLDisplay


class VideoPlayerNvdecGL(_BaseGlPlayer):
    config_desc = "NVDEC decode + TensorRT SR + CUDA-GL zero-copy display"

    def _downscale(self, out: torch.Tensor) -> torch.Tensor:
        """SR output (1,3,H*s,W*s) fp16 RGB [0,255] CUDA -> (3,Ht,Wt) RGB CUDA for GLDisplay.upload,
        bicubic-downscaled to target_size and clamped to [0,255] if one is set.
        The fp16 result is left uncast: upload() fuses the cast to uint8 into its layout copy."""
        if self.target_size is not None:
            out = F.interpolate(out, size=self.target_size, mode="bicubic", align_corners=False)
            out.clamp_(0.0, 255.0)
        return out.squeeze(0)

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
