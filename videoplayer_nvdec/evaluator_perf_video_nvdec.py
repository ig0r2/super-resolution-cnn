import gc
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from videoplayer.scaling import choose_auto_scale
from .decoder import NvDecoder
from .backend import TRTBackendNVDEC

# Reference screen used to derive the display downscale target, so the "full" measurement is
# deterministic and independent of whatever monitor the eval happens to run on.
_REF_SCREEN = (1080, 1920)


class EvaluatorPerfVideoNVDEC:
    """
    Speed evaluation for the NVDEC-decode SR pipeline (videoplayer_nvdec), the NVDEC counterpart
    of videoplayer.evaluator_perf_video_cv2.EvaluatorPerfVideoCV2.

    Unlike the OpenCV evaluator (which feeds a fixed synthetic frame through the model and only
    times inference), this decodes real frames on the GPU via NVDEC and reports a per-stage
    breakdown, so you can see *where* the pipeline is bound:

      - decode_fps : NVDEC decode alone (the hard ceiling the pipeline can never beat)
      - sr_fps     : TensorRT SR alone on a GPU-resident frame (isolates the model)
      - e2e_fps    : decode + SR back to back (what the player's upscale step sustains)
      - full_fps   : decode + SR + GPU bicubic downscale + device->host copy (real display path)

    Small models: sr_fps >> decode_fps, so e2e/full saturate near the decode/overhead floor -> a
    faster model buys little. Large models: sr_fps collapses and dominates e2e/full -> the model
    is the bottleneck. That crossover is exactly what these columns expose.

    Uses TRTBackendNVDEC, so it shares the exports/trt_nvdec engine cache with run_trt_nvdec.py.
    """

    def __init__(self, checkpoint_path, name, video_path, upscale_factor=2,
                 warmup_runs=20, iterations=200):
        self.checkpoint_path = Path(checkpoint_path)
        self.name = name
        self.upscale_factor = upscale_factor
        self.video_path = Path(video_path)
        self.warmup_runs = warmup_runs
        self.iterations = iterations

    def evaluate(self):
        torch.cuda.empty_cache()

        decoder = NvDecoder(str(self.video_path))
        h, w = decoder.height, decoder.width
        n = len(decoder)
        print(f"Video {self.video_path.name}: {w}x{h}, {n} frames, decode target {self.upscale_factor}x")

        # Display downscale target (what the player would show on a 1080p screen).
        decision = choose_auto_scale((h, w), _REF_SCREEN, (self.upscale_factor,))
        target_hw = decision.target_size

        tag = f"{self.name}_{h}x{w}_{self.upscale_factor}x"
        runner = TRTBackendNVDEC(self.checkpoint_path, tag, (h, w), self.upscale_factor)

        def decode(i):
            return decoder.frame(i % n)

        def full(i):
            out = runner(decoder.frame(i % n))
            return self._display(out, target_hw)

        self._open_display(target_hw)
        try:
            fixed = decoder.frame(0)  # GPU-resident frame for the SR-only measurement
            decode_fps = self._time("decode", decode)
            sr_fps = self._time("sr", lambda _i: runner(fixed))
            e2e_fps = self._time("e2e (decode+SR)", lambda i: runner(decoder.frame(i % n)))
            full_fps = self._time("full (decode+SR+display)", full)

            print("-" * 30)
            print(f"decode : {decode_fps}")
            print(f"sr     : {sr_fps}")
            print(f"e2e    : {e2e_fps}")
            print(f"full   : {full_fps}")
            print("-" * 30)

            return {"decode": decode_fps, "sr": sr_fps, "e2e": e2e_fps, "full": full_fps}
        finally:
            self._close_display()
            del runner, decoder
            gc.collect()
            torch.cuda.empty_cache()

    # -- display step (overridden by the CUDA-GL subclass) --------------------------------
    def _open_display(self, target_hw):
        """Hook: set up the display target before the 'full' timing (no-op for the cv2/D2H path)."""

    def _close_display(self):
        """Hook: tear down the display target after timing (no-op for the cv2/D2H path)."""

    def _display(self, out, target_hw):
        """(3,H*s,W*s) uint8 RGB CUDA -> GPU bicubic downscale + BGR + device->host copy to numpy.

        This is the real cv2 display path: the returned numpy array is what cv2.imshow would show,
        and .cpu() forces the copy to complete so the 'full' timing includes the D2H transfer.
        """
        x = out.unsqueeze(0).float()
        x = F.interpolate(x, size=target_hw, mode="bicubic", align_corners=False)
        x = x.clamp(0.0, 255.0).to(torch.uint8).squeeze(0)
        return x[[2, 1, 0]].permute(1, 2, 0).contiguous().cpu().numpy()

    def _time(self, label, fn):
        for i in range(self.warmup_runs):
            fn(i)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for i in range(self.iterations):
            fn(i)
        torch.cuda.synchronize()
        total = time.perf_counter() - start

        avg_ms = (total / self.iterations) * 1000
        fps = 1.0 / (avg_ms / 1000)
        print(f"  {label:28s} {avg_ms:7.2f} ms/frame  ->  {fps:8.1f} FPS")
        return f"{fps:.2f}"
