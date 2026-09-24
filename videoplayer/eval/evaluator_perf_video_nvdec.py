import gc
import time

import torch
import torch.nn.functional as F

from videoplayer.scaling import choose_auto_scale
from videoplayer.decode.decoder import NvDecoder
from videoplayer.backends.nvdec_backend import TRTBackendNVDEC
from ._base import _BaseVideoPerfEvaluator


class EvaluatorPerfVideoNVDEC(_BaseVideoPerfEvaluator):
    """
    Speed evaluation for the NVDEC-decode SR pipeline (players/nvdec.py), the NVDEC counterpart
    of eval/evaluator_perf_video_cv2.py's EvaluatorPerfVideoCV2.

    This decodes real frames on the GPU via NVDEC. A single 'total' loop of `iterations` runs the
    whole pipeline and times each part in-line (with a cuda sync per part, since the stages are
    async on the GPU), so every stage is measured on the same frames in one pass instead of a
    separate loop per stage. All values are average milliseconds per frame:

      - decode  : NVDEC decode alone (the hard floor the pipeline can never beat)
      - sr      : TensorRT SR alone on the decoded frame (isolates the model)
      - display : GPU bicubic downscale + device->host copy alone
      - total   : decode + SR + display (the real display path)

    Small models: sr is small next to decode, so total sits near the decode/overhead floor -> a
    faster model buys little. Large models: sr grows and dominates total -> the model is the
    bottleneck. That crossover is exactly what these columns expose.

    Uses TRTBackendNVDEC, so it shares the exports/trt_nvdec engine cache with run_trt_nvdec.py.
    """

    def evaluate(self):
        torch.cuda.empty_cache()

        decoder = NvDecoder(str(self.video_path))
        h, w = decoder.height, decoder.width
        n = len(decoder)
        print(f"Video {self.video_path.name}: {w}x{h}, {n} frames, decode target {self.upscale_factor}x")

        # Display downscale target (what the player would show on a 1080p screen).
        decision = choose_auto_scale((h, w), self._ref_screen, (self.upscale_factor,))
        target_hw = decision.target_size

        tag = f"{self.name}_{h}x{w}_{self.upscale_factor}x"
        runner = TRTBackendNVDEC(self.checkpoint_path, tag, (h, w), self.upscale_factor)

        def total_step(i):
            self._display(runner(decoder.frame(i % n)), target_hw)

        self._open_display(target_hw)
        try:
            # Warm up the whole pipeline, then time each part inside one 'total' loop. Every stage
            # is async on the GPU, so a cuda sync bounds each part before reading the clock.
            for i in range(self.warmup_runs):
                total_step(i)
            torch.cuda.synchronize()

            totals = {"decode": 0.0, "sr": 0.0, "display": 0.0}
            for i in range(self.iterations):
                t0 = time.perf_counter()
                frame = decoder.frame(i % n)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                out = runner(frame)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                self._display(out, target_hw)
                torch.cuda.synchronize()      # base: D2H .cpu() syncs; GL: D2D upload on default stream
                t3 = time.perf_counter()
                totals["decode"] += t1 - t0
                totals["sr"] += t2 - t1
                totals["display"] += t3 - t2

            return self._summarize(totals)
        finally:
            self._close_display()
            del runner, decoder
            gc.collect()
            torch.cuda.empty_cache()

    # -- display step (overridden by the CUDA-GL subclass) --------------------------------
    def _open_display(self, target_hw):
        """Hook: set up the display target before the 'display' timing (no-op for the cv2/D2H path)."""

    def _close_display(self):
        """Hook: tear down the display target after timing (no-op for the cv2/D2H path)."""

    def _display(self, out, target_hw):
        """(3,H*s,W*s) uint8 RGB CUDA -> GPU bicubic downscale + BGR + device->host copy to numpy.

        This is the real cv2 display path: the returned numpy array is what cv2.imshow would show,
        and .cpu() forces the copy to complete so the 'display' timing includes the D2H transfer.
        """
        x = out.unsqueeze(0).float()
        x = F.interpolate(x, size=target_hw, mode="bicubic", align_corners=False)
        x = x.clamp(0.0, 255.0).to(torch.uint8).squeeze(0)
        return x[[2, 1, 0]].permute(1, 2, 0).contiguous().cpu().numpy()
