import time
from pathlib import Path

import torch

from .export_trt_engine import get_raw_trt_engine, TRTRawRunner
from .. import cache_paths
from .wrapper import VideoWrapperNVDEC, export_onnx_chw


def _log(msg: str):
    print(f"[videoplayer] {msg}")


def _load_model(checkpoint_path, upscale_factor):
    from utils.checkpoints import load_model_from_checkpoint

    _log(f"Loading checkpoint {Path(checkpoint_path).name}")
    t0 = time.perf_counter()
    model, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
    model.upscale_factor = upscale_factor
    _log(f"Checkpoint loaded in {time.perf_counter() - t0:.1f}s")
    return model


def get_onnx(checkpoint_path, onnx_path, input_size, upscale_factor):
    """Return `onnx_path`, exporting the CHW (VideoWrapperNVDEC) ONNX from the checkpoint only if the
    ONNX is missing -- the NVDEC counterpart of videoplayer.backends.get_onnx.

    A cached ONNX means the .pth is never loaded; the checkpoint is only the fallback used to build
    the ONNX the TensorRT engine is then built from.
    """
    if onnx_path.exists():
        _log(f"Reusing cached ONNX {onnx_path.name}")
        return onnx_path

    # Nothing cached to build from and no checkpoint to export from -> nothing we can do.
    if not Path(checkpoint_path).exists():
        raise RuntimeError(f"No cached ONNX ({onnx_path.name}) and no checkpoint ({Path(checkpoint_path).name})")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model(checkpoint_path, upscale_factor)
    _log(f"Exporting CHW ONNX ({input_size[0]}x{input_size[1]}) -> {onnx_path.name} ...")
    t0 = time.perf_counter()
    export_onnx_chw(VideoWrapperNVDEC(model), onnx_path, (input_size[0], input_size[1]))
    _log(f"ONNX export done in {time.perf_counter() - t0:.1f}s")
    return onnx_path


class TRTBackendNVDEC:
    """Raw TensorRT SR backend for the NVDEC-decode pipeline.

    Callable on a (3,H,W) uint8 RGB CUDA tensor (as produced by NvDecoder) and returns a
    (3,H*s,W*s) uint8 RGB CUDA tensor. Everything stays on the GPU; the engine graph itself
    does the normalize/model/denormalize (baked in via VideoWrapperNVDEC at export time).
    """

    def __init__(self, checkpoint_path, tag: str, input_size, upscale_factor: int):
        onnx_path = cache_paths.onnx_nvdec(tag)
        engine_path = cache_paths.engine_nvdec(tag)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer cached artifacts: a cached engine skips everything; otherwise build it from the CHW
        # ONNX, exporting that from the checkpoint only if it isn't cached either.
        if not engine_path.exists():
            get_onnx(checkpoint_path, onnx_path, input_size, upscale_factor)

        _log("Building/loading TensorRT engine (first run for this size/scale can take a while) ...")
        t0 = time.perf_counter()
        engine = get_raw_trt_engine(onnx_path, engine_path)
        _log(f"TensorRT engine ready in {time.perf_counter() - t0:.1f}s")
        self.runner = TRTRawRunner(engine)

        _log("TRTBackendNVDEC ready.")

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        return self.runner(frame)
