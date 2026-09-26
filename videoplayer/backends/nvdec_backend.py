import time

import torch

from .export import get_onnx_video
from .export_trt_engine import get_raw_trt_engine, TRTRawRunner
from .wrappers import OutputLayout, VideoIO
from .. import cache_paths


def _log(msg: str):
    print(f"[videoplayer] {msg}")


class TRTBackendNVDEC:
    """Raw TensorRT SR backend for the NVDEC-decode pipeline.

    Callable on a (3,H,W) uint8 RGB CUDA tensor (as produced by NvDecoder); everything stays on the
    GPU. The engine graph (a VideoWrapper) does the normalize / model / denormalize and the output
    layout: output="bgr" -> (H*s,W*s,3) uint8 BGR for the cv2 display, output="rgb" ->
    (3,H*s,W*s) uint8 RGB for the CUDA-GL display, output="rgb_f16" -> (1,3,H*s,W*s) fp16 RGB
    [0,255] for a GPU downscale before either display.
    """

    def __init__(self, model, input_size, upscale_factor: int, output: OutputLayout):
        checkpoint_path, tag = cache_paths.resolve_model(model, input_size, upscale_factor)
        io = VideoIO("rgb", output)
        onnx_path = cache_paths.onnx_video(tag, io.tag)
        engine_path = cache_paths.engine_video(tag, io.tag)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer cached artifacts: a cached engine skips everything; otherwise build it from the
        # ONNX, exporting that from the checkpoint only if it isn't cached either.
        if not engine_path.exists():
            get_onnx_video(checkpoint_path, onnx_path, input_size, upscale_factor, io)

        _log(f"Building/loading TensorRT engine {engine_path.name} (first run for this size/scale can take a while) ...")
        t0 = time.perf_counter()
        engine = get_raw_trt_engine(onnx_path, engine_path)
        _log(f"TensorRT engine ready in {time.perf_counter() - t0:.1f}s")
        self.runner = TRTRawRunner(engine)

        _log("TRTBackendNVDEC ready.")

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        return self.runner(frame)
