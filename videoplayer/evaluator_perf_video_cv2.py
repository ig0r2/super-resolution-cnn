import gc
import time
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import torch

from videoplayer.backends import TRTBackend, ONNXBackend, NCNNBackend, PT2Backend
from videoplayer.scaling import choose_auto_scale

Runtype: TypeAlias = Literal[
    "tensorrt", "tensorrt-pt2", "ncnn-vulkan",
    "onnxruntime-cuda", "onnxruntime-tensorrt", "onnxruntime-openvino",
    "onnxruntime-directml", "onnxruntime-cpu"]

# Reference screen used to derive the display downscale target, so the "full" measurement is
# deterministic and independent of whatever monitor the eval happens to run on.
_REF_SCREEN = (1080, 1920)


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


class EvaluatorPerfVideoCV2:
    """
    Speed evaluation for the OpenCV-decode SR pipeline (videoplayer/), supporting the tensorrt,
    onnxruntime-* and ncnn-vulkan backends. The CPU-decode counterpart of
    videoplayer_nvdec/evaluator_perf_video_nvdec.py's NVDEC evaluator.

    Real frames are decoded with cv2.VideoCapture (CPU decode + host->device upload happens
    inside the backend), and a per-stage breakdown is reported so you can see where the pipeline
    is bound:

      - decode_fps : cv2 CPU decode alone
      - sr_fps     : backend inference alone on a fixed frame (H2D + model + D2H, as the player pays)
      - e2e_fps    : decode + backend
      - full_fps   : decode + backend + cv2 bicubic downscale (the real display path)

    Small models: sr_fps is high and e2e/full saturate near the decode floor -> a faster model
    buys little. Large models: sr_fps collapses and dominates -> the model is the bottleneck.
    """

    def __init__(self, checkpoint_path, name, video_path, runtype: Runtype, upscale_factor=2,
                 warmup_runs=20, iterations=200):
        self.checkpoint_path = Path(checkpoint_path)
        self.name = name
        self.video_path = Path(video_path)
        self.runtype: Runtype = runtype
        self.upscale_factor = upscale_factor
        self.warmup_runs = warmup_runs
        self.iterations = iterations

    def evaluate(self):
        torch.cuda.empty_cache()

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open {self.video_path}")
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video {self.video_path.name}: {w}x{h}, {n} frames, {self.runtype}")

        decision = choose_auto_scale((h, w), _REF_SCREEN, (self.upscale_factor,))
        target_hw = decision.target_size

        tag = f"{self.name}_{h}x{w}_{self.upscale_factor}x"
        backend = _make_backend(self.runtype, self.checkpoint_path, tag, (h, w), self.upscale_factor)

        def read_next(_i=None):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            return frame

        fixed = read_next()  # one decoded BGR frame for the SR-only measurement

        def full(_i=None):
            out = backend(read_next())
            return cv2.resize(out, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)

        try:
            decode_fps = self._time("decode", read_next)
            sr_fps = self._time("sr", lambda _i: backend(fixed))
            e2e_fps = self._time("e2e (decode+SR)", lambda _i: backend(read_next()))
            full_fps = self._time("full (decode+SR+display)", full)

            print("-" * 30)
            print(f"decode : {decode_fps}")
            print(f"sr     : {sr_fps}")
            print(f"e2e    : {e2e_fps}")
            print(f"full   : {full_fps}")
            print("-" * 30)

            return {"decode": decode_fps, "sr": sr_fps, "e2e": e2e_fps, "full": full_fps}
        finally:
            cap.release()
            del backend
            gc.collect()
            torch.cuda.empty_cache()

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
