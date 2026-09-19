import time

import numpy as np
import torch

from utils.video.wrapper import VideoWrapperCV2
from utils.video.export import export_trt, export_onnx_bare, export_onnx_uint8
from utils.video.export_trt_engine import get_raw_trt_engine, TRTRawRunner
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


def get_onnx(checkpoint_path, onnx_path, input_size, upscale_factor, *, wrap=True):
    """Return `onnx_path`, exporting it from the checkpoint only if the ONNX is missing.
        wrap=True  -> VideoWrapperCV2: for onnx runtime and tensorrt
        wrap=False -> bare model (CHW RGB): for ncnn/pnnx
    """
    if onnx_path.exists():
        _log(f"Reusing cached ONNX {onnx_path.name}")
        return onnx_path

    if not checkpoint_path.exists():
        raise RuntimeError(f"No cached ONNX ({onnx_path.name}) and no checkpoint ({checkpoint_path.name})")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model(checkpoint_path, upscale_factor)
    _log(f"Exporting {'cv2 wrapper' if wrap else 'bare'} ONNX "
         f"({input_size[0]}x{input_size[1]}) -> {onnx_path.name} ...")
    t0 = time.perf_counter()
    if wrap:
        export_onnx_uint8(VideoWrapperCV2(model), onnx_path, (input_size[0], input_size[1]))
    else:
        export_onnx_bare(model, onnx_path, (input_size[0], input_size[1]))
    _log(f"ONNX export done in {time.perf_counter() - t0:.1f}s")
    return onnx_path


class TRTBackend:
    """Raw TensorRT engine backend. Callable on a BGR uint8 numpy frame (H,W,3)."""

    def __init__(self, checkpoint_path, tag: str, input_size, upscale_factor: int):
        onnx_path = cache_paths.onnx_cv2(tag)
        engine_path = cache_paths.engine_cv2(tag)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer cached artifacts: a cached engine skips everything; otherwise build it from the
        # shared cv2 ONNX, exporting that from the checkpoint only if it isn't cached either.
        if not engine_path.exists():
            get_onnx(checkpoint_path, onnx_path, input_size, upscale_factor, wrap=True)

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


class PT2Backend:
    """torch_tensorrt (.pt2) backend. Callable on a BGR uint8 numpy frame (H,W,3).

    Unlike TRTBackend, which builds a raw TensorRT *engine* from ONNX, this compiles the model
    with torch_tensorrt and serialises a .pt2. It's the safer choice for LARGE models: building a
    raw TensorRT engine can run out of VRAM (TensorRT needs a big scratch/workspace pool plus
    tactic-profiling memory at build time, on top of the weights), and for the biggest models that
    build simply OOMs. The torch_tensorrt path builds more conservatively and keeps working where
    the engine build dies -- so reach for tensorrt-pt2 whenever `tensorrt` fails to build. For
    small/medium models the raw engine (TRTBackend) is a bit faster, so prefer it when it fits.
    """

    def __init__(self, checkpoint_path, tag: str, input_size, upscale_factor: int):
        import torch_tensorrt  # noqa: F401  (registers the ops needed to load the .pt2)

        pt2_path = cache_paths.pt2_cv2(tag)
        pt2_path.parent.mkdir(parents=True, exist_ok=True)

        # No ONNX step: torch_tensorrt compiles the torch model directly, so a cache miss needs the
        # checkpoint. A cached .pt2 skips both the checkpoint load and the (slow) compile.
        if not pt2_path.exists() and not checkpoint_path.exists():
            raise RuntimeError(f"No cached .pt2 ({pt2_path.name}) and no checkpoint ({checkpoint_path.name})")

        if not pt2_path.exists():
            model = _load_model(checkpoint_path, upscale_factor)
            model.half()
            _log(f"Compiling torch_tensorrt .pt2 ({input_size[0]}x{input_size[1]}) -> {pt2_path.name} ...")
            t0 = time.perf_counter()
            export_trt(VideoWrapperCV2(model), pt2_path, (input_size[0], input_size[1], 3))
            _log(f".pt2 compile done in {time.perf_counter() - t0:.1f}s")
        else:
            _log(f"Reusing cached .pt2 model {pt2_path.name}")

        self.model = torch.export.load(str(pt2_path)).module().cuda()

        # Pinned staging buffer for async H2D uploads, allocated lazily on the first frame.
        self._staging = None

        _log("PT2Backend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self._staging is None or tuple(self._staging.shape) != frame.shape:
            self._staging = torch.empty(frame.shape, dtype=torch.uint8, pin_memory=True)
        # The compiled module expects fp16 input directly (the raw runner casts on-GPU instead).
        frame_gpu = self._staging.copy_(torch.from_numpy(frame)).cuda(non_blocking=True).half()
        return self.model(frame_gpu).cpu().numpy()


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

        onnx_path = get_onnx(checkpoint_path, cache_paths.onnx_cv2(tag), input_size, upscale_factor, wrap=True)

        _log(f"Creating onnxruntime session (provider={provider}) ...")
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(onnx_path), providers=self.PROVIDER_MAP[provider])
        _log(f"onnxruntime session ready in {time.perf_counter() - t0:.1f}s")
        _log("ONNXBackend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"input": frame})[0]


class NCNNBackend:
    """ncnn-Vulkan backend. Callable on a BGR uint8 numpy frame (H,W,3)."""

    def __init__(self, checkpoint_path, tag: str, input_size, upscale_factor: int):
        param_path, bin_path = cache_paths.ncnn_paths(tag)
        param_path.parent.mkdir(parents=True, exist_ok=True)

        # ncnn files first, otherwise convert from .onnx, exporting from checkpoint if no .onnx
        if not param_path.exists():
            onnx_path = get_onnx(checkpoint_path, cache_paths.onnx(tag), input_size, upscale_factor, wrap=False)
            _log(f"Converting ncnn from ONNX ({input_size[0]}x{input_size[1]}) -> {param_path.name} ...")
            t0 = time.perf_counter()
            export_ncnn(onnx_path, param_path.parent, tag, (input_size[0], input_size[1]))
            _log(f"ncnn conversion done in {time.perf_counter() - t0:.1f}s")
        else:
            _log(f"Reusing cached ncnn model {param_path.name}")

        self.runner = NCNNRunner(param_path, bin_path)

        _log("NCNNBackend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.runner(frame)
