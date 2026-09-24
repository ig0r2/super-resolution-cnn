import gc
import time
from typing import Literal, TypeAlias

import cv2
import torch

from videoplayer.backends.cv2_backends import TRTBackend, ONNXBackend, NCNNBackend, PT2Backend
from videoplayer.scaling import choose_auto_scale
from ._base import _BaseVideoPerfEvaluator

Runtype: TypeAlias = Literal[
    "tensorrt", "tensorrt-pt2", "ncnn-vulkan",
    "onnxruntime-cuda", "onnxruntime-tensorrt", "onnxruntime-openvino",
    "onnxruntime-directml", "onnxruntime-cpu"]


def _make_backend(runtype: Runtype, checkpoint_path, tag, input_size, upscale_factor):
    """Build the videoplayer backend for a runtype (same classes the demo player uses)."""
    if runtype == "tensorrt":
        return TRTBackend(checkpoint_path, tag, input_size, upscale_factor)
    if runtype == "tensorrt-pt2":
        # torch_tensorrt .pt2: safer for large models where the raw engine build OOMs (see PT2Backend).
        return PT2Backend(checkpoint_path, tag, input_size, upscale_factor)
    if runtype == "ncnn-vulkan":
        return NCNNBackend(checkpoint_path, tag, input_size, upscale_factor)
    if runtype.startswith("onnxruntime-"):
        provider = runtype.split("-", 1)[1]  # cuda / tensorrt / openvino / directml / cpu
        return ONNXBackend(checkpoint_path, tag, input_size, upscale_factor, provider=provider)
    raise ValueError(f"Unknown runtype: {runtype}")


class EvaluatorPerfVideoCV2(_BaseVideoPerfEvaluator):
    """
    Speed evaluation for the OpenCV-decode SR pipeline (players/cv2_player.py), supporting the
    tensorrt, onnxruntime-* and ncnn-vulkan backends. The CPU-decode counterpart of
    eval/evaluator_perf_video_nvdec.py's NVDEC evaluator.

    Real frames are decoded with cv2.VideoCapture (CPU decode + host->device upload happens
    inside the backend). All values are average milliseconds per frame:

      - decode  : cv2 CPU decode alone
      - sr      : backend inference alone (H2D + model + D2H, as the player pays)
      - display : cv2 bicubic downscale to the display size alone
      - total   : decode + sr + display (the real display path)
    """

    def __init__(self, checkpoint_path, name, video_path, runtype: Runtype, upscale_factor=2,
                 warmup_runs=20, iterations=200):
        super().__init__(checkpoint_path, name, video_path, upscale_factor, warmup_runs, iterations)
        self.runtype: Runtype = runtype

    def evaluate(self):
        torch.cuda.empty_cache()

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open {self.video_path}")
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video {self.video_path.name}: {w}x{h}, {n} frames, {self.runtype}")

        decision = choose_auto_scale((h, w), self._ref_screen, (self.upscale_factor,))
        target_hw = decision.target_size

        tag = f"{self.name}_{h}x{w}_{self.upscale_factor}x"
        backend = _make_backend(self.runtype, self.checkpoint_path, tag, (h, w), self.upscale_factor)

        def read_next(_i=None):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
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
            cap.release()
            del backend
            gc.collect()
            torch.cuda.empty_cache()
