import time
from pathlib import Path

import torch

from utils.video.export_trt_engine import get_raw_trt_engine, TRTRawRunner
from videoplayer import cache_paths
from .wrapper import VideoWrapperNVDEC, export_onnx_chw


def _log(msg: str):
    print(f"[videoplayer_nvdec] {msg}")


def _load_model(checkpoint_path, upscale_factor):
    from utils.checkpoints import load_model_from_checkpoint

    _log(f"Loading checkpoint {Path(checkpoint_path).name}")
    t0 = time.perf_counter()
    model, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
    model.upscale_factor = upscale_factor
    _log(f"Checkpoint loaded in {time.perf_counter() - t0:.1f}s")
    return model


class TRTBackendNVDEC:
    """Raw TensorRT SR backend for the NVDEC-decode pipeline.

    Callable on a (3,H,W) uint8 RGB CUDA tensor (as produced by NvDecoder) and returns a
    (3,H*s,W*s) uint8 RGB CUDA tensor. Everything stays on the GPU; the engine graph itself
    does the normalize/model/denormalize (baked in via VideoWrapperNVDEC at export time).
    """

    def __init__(self, checkpoint_path, tag: str, input_size, upscale_factor: int):
        onnx_path = cache_paths.onnx_nvdec(tag)
        engine_path = cache_paths.engine_nvdec(tag)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        # The checkpoint is only needed to export the ONNX; a cached engine skips it, and a cached
        # ONNX lets us build the engine without it.
        if not engine_path.exists() and not onnx_path.exists():
            model = _load_model(checkpoint_path, upscale_factor)
            _log(f"Exporting ONNX ({input_size[0]}x{input_size[1]}, CHW) -> {onnx_path.name} ...")
            t0 = time.perf_counter()
            export_onnx_chw(VideoWrapperNVDEC(model), onnx_path, (input_size[0], input_size[1]))
            _log(f"ONNX export done in {time.perf_counter() - t0:.1f}s")

        _log("Building/loading TensorRT engine (first run for this size/scale can take a while) ...")
        t0 = time.perf_counter()
        engine = get_raw_trt_engine(onnx_path, engine_path)
        _log(f"TensorRT engine ready in {time.perf_counter() - t0:.1f}s")
        self.runner = TRTRawRunner(engine)

        _log("TRTBackendNVDEC ready.")

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        return self.runner(frame)
