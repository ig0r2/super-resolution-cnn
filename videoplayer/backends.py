import time
from pathlib import Path

import numpy as np
import torch

from utils.video.evaluator_perf_video import VideoWrapperCV2
from utils.video.export import export_onnx
from utils.video.export_trt_engine import export_onnx_raw, get_raw_trt_engine, TRTRawRunner
from utils.video.model_utils import TileProcessor, TileProcessorTorch


def _log(msg: str):
    print(f"[videoplayer] {msg}")


def _load_model(checkpoint_path, upscale_factor):
    from utils.checkpoints import load_model_from_checkpoint

    _log(f"Loading checkpoint {checkpoint_path.name}")
    t0 = time.perf_counter()
    model, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
    model.upscale_factor = upscale_factor
    _log(f"Checkpoint loaded in {time.perf_counter() - t0:.1f}s")
    return model


class TRTBackend:
    """Raw TensorRT engine backend. Callable on a BGR uint8 numpy frame (H,W,3)."""

    def __init__(self, checkpoint_path: Path, cache_dir: Path, tag: str, input_size, upscale_factor: int,
                 tiled: bool = False, tile_size: int = 256):
        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_dir / f"{tag}.onnx"
        engine_path = cache_dir / f"{tag}.engine"

        if not engine_path.exists():
            model = _load_model(checkpoint_path, upscale_factor)
            _log(f"Exporting ONNX ({input_size[0]}x{input_size[1]}) -> {onnx_path.name} ...")
            t0 = time.perf_counter()
            export_onnx_raw(VideoWrapperCV2(model), onnx_path, (input_size[0], input_size[1]))
            _log(f"ONNX export done in {time.perf_counter() - t0:.1f}s")
        else:
            _log(f"Reusing cached engine {engine_path.name}")

        _log("Building/loading TensorRT engine (first run for this size/scale can take a while) ...")
        t0 = time.perf_counter()
        engine = get_raw_trt_engine(onnx_path, engine_path)
        _log(f"TensorRT engine ready in {time.perf_counter() - t0:.1f}s")
        self.runner = TRTRawRunner(engine)

        self.tiled = tiled
        if tiled:
            self.tile_processor = TileProcessorTorch(upscale_factor=upscale_factor, tile_size=tile_size, overlap=8)

        # Pinned staging buffer for async H2D uploads, allocated lazily on the first frame (the
        # full-frame size isn't known here in tiled mode). The runner casts uint8 -> fp16 on GPU.
        self._staging = None

        _log("TRTBackend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self._staging is None or tuple(self._staging.shape) != frame.shape:
            self._staging = torch.empty(frame.shape, dtype=torch.uint8, pin_memory=True)
        frame_gpu = self._staging.copy_(torch.from_numpy(frame)).cuda(non_blocking=True)
        if self.tiled:
            out = self.tile_processor.process_frame(frame_gpu, self.runner)
        else:
            out = self.runner(frame_gpu)
        return out.cpu().numpy()


class ONNXBackend:
    """onnxruntime backend. Callable on a BGR uint8 numpy frame (H,W,3)."""

    PROVIDER_MAP = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": [("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CUDAExecutionProvider"],
        "directml": ["DmlExecutionProvider"],
        "openvino": [("OpenVINOExecutionProvider", {"device_type": "GPU", "precision": "FP16"})],
        "cpu": ["CPUExecutionProvider"],
    }

    def __init__(self, checkpoint_path: Path, cache_dir: Path, tag: str, input_size, upscale_factor: int,
                 provider: str = "cuda", tiled: bool = False, tile_size: int = 256):
        import onnxruntime as ort

        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_dir / f"{tag}.onnx"

        if not onnx_path.exists():
            model = _load_model(checkpoint_path, upscale_factor)
            model.half()
            _log(f"Exporting ONNX ({input_size[0]}x{input_size[1]}) -> {onnx_path.name} ...")
            t0 = time.perf_counter()
            export_onnx(VideoWrapperCV2(model), onnx_path, (input_size[0], input_size[1], 3))
            _log(f"ONNX export done in {time.perf_counter() - t0:.1f}s")
        else:
            _log(f"Reusing cached ONNX model {onnx_path.name}")

        _log(f"Creating onnxruntime session (provider={provider}) ...")
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(onnx_path), providers=self.PROVIDER_MAP[provider])
        _log(f"onnxruntime session ready in {time.perf_counter() - t0:.1f}s")
        self.dtype = np.float16

        self.tiled = tiled
        if tiled:
            self.tile_processor = TileProcessor(upscale_factor=upscale_factor, tile_size=tile_size, overlap=8)

        _log("ONNXBackend ready.")

    def _infer(self, tile: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"input": tile.astype(self.dtype)})[0]

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.tiled:
            return self.tile_processor.process_frame(frame, self._infer)
        return self._infer(frame)
