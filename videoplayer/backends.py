import time

import numpy as np
import torch

from utils.video.evaluator_perf_video import VideoWrapperCV2
from utils.video.export_trt_engine import export_onnx_raw, get_raw_trt_engine, TRTRawRunner
from utils.video.export_ncnn import export_ncnn, NCNNRunner
from videoplayer import cache_paths


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

    def __init__(self, checkpoint_path, tag: str, input_size, upscale_factor: int):
        onnx_path = cache_paths.onnx_cv2(tag)
        engine_path = cache_paths.engine_cv2(tag)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        # The checkpoint is only needed to export the (shared) ONNX; a cached engine skips it, and
        # a cached ONNX (e.g. built by the onnxruntime backend) lets us build the engine without it.
        if not engine_path.exists() and not onnx_path.exists():
            model = _load_model(checkpoint_path, upscale_factor)
            _log(f"Exporting ONNX ({input_size[0]}x{input_size[1]}) -> {onnx_path.name} ...")
            t0 = time.perf_counter()
            export_onnx_raw(VideoWrapperCV2(model), onnx_path, (input_size[0], input_size[1]))
            _log(f"ONNX export done in {time.perf_counter() - t0:.1f}s")

        _log("Building/loading TensorRT engine (first run for this size/scale can take a while) ...")
        t0 = time.perf_counter()
        engine = get_raw_trt_engine(onnx_path, engine_path)
        _log(f"TensorRT engine ready in {time.perf_counter() - t0:.1f}s")
        self.runner = TRTRawRunner(engine)

        # Pinned staging buffer for async H2D uploads, allocated lazily on the first frame.
        # The runner casts uint8 -> fp16 on GPU.
        self._staging = None

        _log("TRTBackend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self._staging is None or tuple(self._staging.shape) != frame.shape:
            self._staging = torch.empty(frame.shape, dtype=torch.uint8, pin_memory=True)
        frame_gpu = self._staging.copy_(torch.from_numpy(frame)).cuda(non_blocking=True)
        return self.runner(frame_gpu).cpu().numpy()


class ONNXBackend:
    """onnxruntime backend. Callable on a BGR uint8 numpy frame (H,W,3)."""

    PROVIDER_MAP = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": [("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CUDAExecutionProvider"],
        "directml": ["DmlExecutionProvider"],
        "openvino": [("OpenVINOExecutionProvider", {"device_type": "GPU", "precision": "FP16"})],
        "cpu": ["CPUExecutionProvider"],
    }

    def __init__(self, checkpoint_path, tag: str, input_size, upscale_factor: int, provider: str = "cuda"):
        import onnxruntime as ort

        # Shared cv2 ONNX (same file the TensorRT backend builds its engine from).
        onnx_path = cache_paths.onnx_cv2(tag)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        if not onnx_path.exists():
            model = _load_model(checkpoint_path, upscale_factor)
            _log(f"Exporting ONNX ({input_size[0]}x{input_size[1]}) -> {onnx_path.name} ...")
            t0 = time.perf_counter()
            export_onnx_raw(VideoWrapperCV2(model), onnx_path, (input_size[0], input_size[1]))
            _log(f"ONNX export done in {time.perf_counter() - t0:.1f}s")
        else:
            _log(f"Reusing cached ONNX model {onnx_path.name}")

        _log(f"Creating onnxruntime session (provider={provider}) ...")
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(onnx_path), providers=self.PROVIDER_MAP[provider])
        _log(f"onnxruntime session ready in {time.perf_counter() - t0:.1f}s")
        self.dtype = np.float16

        _log("ONNXBackend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"input": frame.astype(self.dtype)})[0]


class NCNNBackend:
    """ncnn-Vulkan backend. Callable on a BGR uint8 numpy frame (H,W,3)."""

    def __init__(self, checkpoint_path, tag: str, input_size, upscale_factor: int):
        param_path, bin_path = cache_paths.ncnn_paths(tag)
        param_path.parent.mkdir(parents=True, exist_ok=True)

        if not param_path.exists():
            model = _load_model(checkpoint_path, upscale_factor)
            _log(f"Converting to ncnn ({input_size[0]}x{input_size[1]}) -> {param_path.name} ...")
            t0 = time.perf_counter()
            export_ncnn(model, param_path.parent, tag, (input_size[0], input_size[1]))
            _log(f"ncnn conversion done in {time.perf_counter() - t0:.1f}s")
        else:
            _log(f"Reusing cached ncnn model {param_path.name}")

        self.runner = NCNNRunner(param_path, bin_path)

        _log("NCNNBackend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.runner(frame)
