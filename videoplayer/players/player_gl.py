"""
Pure-OpenGL SR video player.

_BaseGlPlayer provides decode / background reader / audio / seeking and the glfw display loop;
this player swaps the ML upscale step for a GLUpscaler that runs the model as GLSL shader passes,
with no torch conv, TensorRT, or ONNX runtime involved. Two input paths:

  decoder="nvdec"       : NVDEC frame -> input GL texture via CUDA interop (all on the GPU; NVIDIA only)
  decoder="pyav" / "cv2": CPU-decoded BGR frame -> input GL texture via glTexSubImage2D (no CUDA,
                          runs on any OpenGL 3.3 GPU: AMD, Intel, NVIDIA)
"""

from typing import Optional, Tuple

from OpenGL import GL

from .base_gl import _BaseGlPlayer
from ..decode import Decoder
from ..backends.gl_build import build_engine
from ..cache_paths import resolve_model


class VideoPlayerGL(_BaseGlPlayer):
    """Takes the model (checkpoint name or .pth path) directly: the scale is picked for the screen
    on construction, and the shader graph is compiled (which also opens the window) on play().
    chunk_size is the number of input groups per conv chunk (must keep bound textures <=
    GL_MAX_TEXTURE_IMAGE_UNITS)."""

    def __init__(self, video_path, model, decoder: Decoder = "nvdec", candidate_scales=(2, 3, 4),
                 chunk_size: int = 8, target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True):
        self.config_desc = f"{decoder} decode + GLSL-shader SR (pure OpenGL, no ML runtime)"
        super().__init__(video_path, decoder=decoder, target_size=target_size,
                         enable_audio=enable_audio, start_fullscreen=start_fullscreen)
        self.input = "cuda" if decoder == "nvdec" else "host"
        self.model = model
        self.chunk_size = chunk_size
        self.scale = self.configure_scale(candidate_scales)
        self.engine = None

    def _gl_ready(self) -> bool:
        return True

    def _gl_open(self, on_key):
        lr_h, lr_w = self.size
        # Window opens at the display target if we have one, else the native SR output size.
        win_h, win_w = self.target_size if self.target_size is not None else (lr_h * self.scale, lr_w * self.scale)

        checkpoint_path, _ = resolve_model(self.model, (lr_h, lr_w), self.scale)
        print(f"[videoplayer] Loading checkpoint {self.model} and compiling GLSL shader graph ...")
        self.engine, meta = build_engine(checkpoint_path, self.scale, lr_h, lr_w,
                                         win_w=win_w, win_h=win_h, title=self.window_name,
                                         fullscreen=self.fullscreen, chunk_size=self.chunk_size, input=self.input)
        print(f"[videoplayer] {meta['num_passes']} render passes "
              f"(num_blocks={meta['num_blocks']}, nf={meta['nf']}, {self.scale}x)")

        self.engine.set_key_callback(on_key)
        return self.engine

    def _gl_infer(self, frame):
        self.engine.infer(frame)

    def _gl_sync(self):
        # The shader passes run on the GL queue, so wait for them with glFinish on both paths. With
        # only a CUDA sync they stay in flight, and the next frame's cudaGraphicsMapResources then
        # waits for them inside the CUDA driver, which stalls the reader thread's NVDEC decode: with
        # a model slower than the video, playback itself slowed down instead of skipping frames.
        if self.input == "cuda":
            super()._gl_sync()
        GL.glFinish()

    def _gl_present(self, title: str, have_frame: bool):
        if have_frame:
            self.engine.present(title)
