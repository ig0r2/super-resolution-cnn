"""
Pure-OpenGL SR video player.

Reuses VideoPlayerNvdecCV2 for NVDEC decode / background reader / audio / seeking and _BaseGlPlayer
for the glfw display loop, and swaps the ML upscale step for a GLUpscaler that runs the model as
GLSL shader passes. Everything stays on the GPU: NVDEC frame -> input GL texture (CUDA interop) ->
shader graph -> window, with no torch conv, TensorRT, or ONNX runtime involved.
"""

from .base_gl import _BaseGlPlayer
from .nvdec import VideoPlayerNvdecCV2
from ..backends.gl_engine import GLUpscaler


class VideoPlayerGL(_BaseGlPlayer, VideoPlayerNvdecCV2):
    config_desc = "NVDEC decode + GLSL-shader SR (pure OpenGL, no ML runtime)"

    def set_engine(self, engine: GLUpscaler):
        self.engine = engine
        return self

    def _gl_ready(self) -> bool:
        return getattr(self, "engine", None) is not None

    def _gl_open(self, on_key):
        self.engine.set_key_callback(on_key)
        return self.engine

    def _gl_infer(self, frame):
        self.engine.infer(frame)

    def _gl_present(self, title: str, have_frame: bool):
        if have_frame:
            self.engine.present(title)
