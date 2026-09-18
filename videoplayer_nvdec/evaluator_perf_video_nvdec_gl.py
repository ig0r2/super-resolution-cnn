import torch
import torch.nn.functional as F

from .evaluator_perf_video_nvdec import EvaluatorPerfVideoNVDEC
from .gl_display import GLDisplay


class EvaluatorPerfVideoNVDECGL(EvaluatorPerfVideoNVDEC):
    """
    CUDA-GL display variant of EvaluatorPerfVideoNVDEC (runtype "tensorrt-nvdec-gl").

    decode / sr / e2e are measured identically to the base evaluator; only the 'full' stage
    differs, and it is kept a fair analog of the base one. The base 'full' stops at .cpu().numpy()
    -- i.e. it measures getting the frame off the GPU to where cv2.imshow needs it, NOT the imshow
    blit itself. The zero-copy analog is the same handoff with the D2H PCIe copy replaced by a
    device->device copy into a CUDA-registered GL texture. So both columns measure "SR + downscale
    + deliver the frame to the display surface," and their difference isolates exactly the
    transfer saving (D2H copy vs D2D texture upload); neither includes the final present/blit.

    We deliberately do NOT render/swap/glFinish per frame here: that would force a per-frame CPU-GPU
    stall and measure present latency instead of throughput, and would be asymmetric with the base
    (which never presents). The upload's cudaMemcpy runs on the default stream, so the once-per-run
    torch.cuda.synchronize() in _time captures it. A hidden (offscreen) GL context is still needed
    because the texture must be a real GL object registered with CUDA; visible=False just avoids
    popping a window per model during a sweep.
    """

    def _open_display(self, target_hw):
        h, w = target_hw
        self._disp = GLDisplay(w, h, title="eval", visible=False)

    def _close_display(self):
        disp = getattr(self, "_disp", None)
        if disp is not None:
            disp.close()
            self._disp = None

    def _display(self, out, target_hw):
        # (3,H*s,W*s) uint8 RGB CUDA -> GPU bicubic downscale to (3,Ht,Wt) RGB (no BGR swap needed),
        # then device->device copy into the GL texture (the zero-copy analog of the base D2H copy).
        x = out.unsqueeze(0).float()
        x = F.interpolate(x, size=target_hw, mode="bicubic", align_corners=False)
        x = x.clamp(0.0, 255.0).to(torch.uint8).squeeze(0)
        self._disp.upload(x)
        return None
