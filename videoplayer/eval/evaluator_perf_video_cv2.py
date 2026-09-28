import gc
import time

import cv2
import torch

from videoplayer.backends.cv2_backends import Runtype, make_backend
from videoplayer.players.player_cv2 import Decoder, open_cpu_decoder
from videoplayer.scaling import choose_auto_scale
from ._base import _BaseVideoPerfEvaluator


class EvaluatorPerfVideoCV2(_BaseVideoPerfEvaluator):
    """
    Speed evaluation for the CPU-decode SR pipeline with cv2 display (players/player_cv2.py),
    supporting the tensorrt, tensorrt-pt2, onnxruntime-* and ncnn-vulkan backends. The CPU-decode
    counterpart of eval/evaluator_perf_video_nvdec.py's NVDEC evaluator.

    Real frames are decoded on the CPU with `decoder` ("pyav": PyAV, "cv2": cv2.VideoCapture; both
    (H,W,3) BGR, so both use the same bgr -> bgr engine); the host->device upload happens inside the
    backend. All values are average milliseconds per frame:

      - decode  : CPU decode (incl. YUV->BGR conversion) alone
      - sr      : backend inference alone (H2D + model + D2H, as the player pays)
      - display : cv2 bicubic downscale to the display size alone
      - total   : decode + sr + display (the real display path)
    """

    def __init__(self, checkpoint_path, video_path, runtype: Runtype, decoder: Decoder = "pyav",
                 upscale_factor=2, warmup_runs=20, iterations=200):
        super().__init__(checkpoint_path, video_path, upscale_factor, warmup_runs, iterations)
        self.runtype: Runtype = runtype
        self.decoder: Decoder = decoder

    def evaluate(self):
        torch.cuda.empty_cache()

        decoder = open_cpu_decoder(self.video_path, self.decoder)
        h, w, n = decoder.height, decoder.width, len(decoder)
        print(f"Video {self.video_path.name}: {w}x{h}, {n} frames, {self.runtype}, {self.decoder}")

        decision = choose_auto_scale((h, w), self._ref_screen, (self.upscale_factor,))
        target_hw = decision.target_size

        backend = make_backend(self.runtype, self.checkpoint_path, (h, w), self.upscale_factor)

        cursor = [0]

        def read_next():
            # Sequential decode; wrapping to frame 0 at the end is a (rare) seek.
            if cursor[0] >= len(decoder):
                cursor[0] = 0
            frame = decoder.frame(cursor[0])
            cursor[0] += 1
            return frame

        def downscale(out):
            return cv2.resize(out, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)

        def total_step():
            downscale(backend(read_next()))

        try:
            for _ in range(self.warmup_runs):
                total_step()
            torch.cuda.synchronize()

            totals = {"decode": 0.0, "sr": 0.0, "display": 0.0}
            for _ in range(self.iterations):
                t0 = time.perf_counter()
                frame = read_next()
                t1 = time.perf_counter()
                out = backend(frame)
                t2 = time.perf_counter()
                downscale(out)
                t3 = time.perf_counter()
                totals["decode"] += t1 - t0
                totals["sr"] += t2 - t1
                totals["display"] += t3 - t2

            return self._summarize(totals)
        finally:
            decoder.close()
            del backend
            gc.collect()
            torch.cuda.empty_cache()
