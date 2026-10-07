import torch.nn.functional as F
from OpenGL import GL

from .evaluator_perf_video_nvdec import EvaluatorPerfVideoNVDEC
from videoplayer.backends.gl_display import GLDisplay


class EvaluatorPerfVideoNVDECGL(EvaluatorPerfVideoNVDEC):
    """
    CUDA-GL display variant of EvaluatorPerfVideoNVDEC (runtype "tensorrt-nvdec-gl").

    decode / sr are measured identically to the base evaluator; only the 'display' stage (and thus
    'total') differs, and it is kept a fair analog of the base one. The base 'display' covers the
    D2H copy + cv2.imshow + waitKeyEx (where the window repaint happens). The zero-copy analog is
    the same frame reaching a visible window with the D2H copy replaced by a device->device copy
    into a CUDA-registered GL texture, followed by the draw + buffer swap + event poll the player
    does every frame (GLDisplay.render / poll). glFinish bounds the GL side, so the timing includes
    the GPU work of the draw, not just its submission. The window is visible and windowed at the
    display target size (vsync off, see GLDisplay), never fullscreen.
    """

    def _open_display(self, target_hw):
        h, w = target_hw
        self._disp = GLDisplay(w, h, title=self._window_name, fullscreen=False, visible=True)

    def _close_display(self):
        disp = getattr(self, "_disp", None)
        if disp is not None:
            disp.close()
            self._disp = None

    def _display(self, out, target_hw):
        # (1,3,H*s,W*s) fp16 RGB CUDA -> GPU bicubic downscale to (3,Ht,Wt) fp16 RGB clamped to
        # [0,255] (no BGR swap needed), then upload() casts to uint8 in its layout copy and does the
        # device->device copy into the GL texture (the zero-copy analog of the D2H copy). Same path
        # as VideoPlayerNvdecGL._downscale + _gl_present.
        x = F.interpolate(out, size=target_hw, mode="bicubic", align_corners=False)
        self._disp.upload(x.clamp_(0.0, 255.0).squeeze(0))
        self._disp.render()   # draw + swap_buffers
        self._disp.poll()
        GL.glFinish()
        return None
