"""
SR backends for the cv2-display player (players/player_cv2.py): each is callable on the decoder's
(H,W,3) uint8 BGR numpy frame on the host (PyAV or cv2) and returns a BGR uint8 numpy frame
(H*s,W*s,3) ready for cv2.

Every backend takes `model` (checkpoint name under checkpoints/ or a .pth path), the input frame
size and the scale; the checkpoint path and the cache tag come from cache_paths.resolve_model.

TensorRT / torch_tensorrt / onnxruntime run a VideoWrapper export, so the layout conversion and the
BGR output happen inside the graph. ncnn runs the model with only *255/clamp appended (export_onnx_ncnn)
and does the rest of the pre/post-processing itself.
"""

import time
from typing import Literal

import numpy as np
import torch

from .export import export_trt, get_onnx_video, load_model
from .export_trt_engine import get_raw_trt_engine, TRTRawRunner
from .wrappers import VideoIO
from .. import cache_paths

Runtype = Literal["tensorrt", "tensorrt-pt2", "ncnn-vulkan",
"onnxruntime-cuda", "onnxruntime-tensorrt", "onnxruntime-openvino",
"onnxruntime-directml", "onnxruntime-cpu"]


def _log(msg: str):
    print(f"[videoplayer] {msg}")


# Both CPU decoders give (H,W,3) BGR frames and cv2 shows BGR, so every wrapped export is bgr -> bgr.
_IO = VideoIO("bgr", "bgr")


class _PinnedIO:
    """Pinned host staging buffers for async H2D upload of the input and D2H download of the
    output, allocated lazily on the first frame. The returned array is a view into the pinned
    output buffer, which is reused across calls (consume it before the next call)."""

    def __init__(self):
        self._in = None
        self._out = None

    def upload(self, frame: np.ndarray) -> torch.Tensor:
        if self._in is None or tuple(self._in.shape) != frame.shape:
            self._in = torch.empty(frame.shape, dtype=torch.uint8, pin_memory=True)
        return self._in.copy_(torch.from_numpy(frame)).cuda(non_blocking=True)

    def download(self, out_gpu) -> np.ndarray:
        if self._out is None or self._out.shape != out_gpu.shape:
            self._out = torch.empty(out_gpu.shape, dtype=out_gpu.dtype, pin_memory=True)
        self._out.copy_(out_gpu, non_blocking=True)
        torch.cuda.synchronize()  # finish the async D2H before the numpy view is read
        return self._out.numpy()


class TRTBackend:
    """Raw TensorRT engine backend built from the bgr -> bgr VideoWrapper ONNX."""

    def __init__(self, model, input_size, upscale_factor: int):
        checkpoint_path, tag = cache_paths.resolve_model(model, input_size, upscale_factor)
        onnx_path = cache_paths.onnx_video(tag, _IO.tag)
        engine_path = cache_paths.engine_video(tag, _IO.tag)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        # Prefer cached artifacts: a cached engine skips everything; otherwise build it from the
        # ONNX, exporting that from the checkpoint only if it isn't cached either.
        if not engine_path.exists():
            get_onnx_video(checkpoint_path, onnx_path, input_size, upscale_factor, _IO)

        _log(
            f"Building/loading TensorRT engine {engine_path.name} (first run for this size/scale can take a while) ...")
        t0 = time.perf_counter()
        engine = get_raw_trt_engine(onnx_path, engine_path)
        _log(f"TensorRT engine ready in {time.perf_counter() - t0:.1f}s")
        self.runner = TRTRawRunner(engine)
        self._io = _PinnedIO()  # the runner casts uint8 -> engine dtype on the GPU
        _log("TRTBackend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self._io.download(self.runner(self._io.upload(frame)))


class PT2Backend:
    """torch_tensorrt (.pt2) backend of the bgr -> bgr VideoWrapper.

    Unlike TRTBackend, which builds a raw TensorRT *engine* from ONNX, this compiles the model
    with torch_tensorrt and serialises a .pt2. It's the safer choice for LARGE models: building a
    raw TensorRT engine can run out of VRAM (TensorRT needs a big scratch/workspace pool plus
    tactic-profiling memory at build time, on top of the weights), and for the biggest models that
    build simply OOMs. The torch_tensorrt path builds more conservatively and keeps working where
    the engine build dies -- so reach for tensorrt-pt2 whenever `tensorrt` fails to build. For
    small/medium models the raw engine (TRTBackend) is a bit faster, so prefer it when it fits.
    """

    def __init__(self, model, input_size, upscale_factor: int):
        import torch_tensorrt  # noqa: F401  (registers the ops needed to load the .pt2)
        checkpoint_path, tag = cache_paths.resolve_model(model, input_size, upscale_factor)

        pt2_path = cache_paths.pt2_video(tag, _IO.tag)
        pt2_path.parent.mkdir(parents=True, exist_ok=True)

        # No ONNX step: torch_tensorrt compiles the torch model directly, so a cache miss needs the
        # checkpoint. A cached .pt2 skips both the checkpoint load and the (slow) compile.
        if not pt2_path.exists() and not checkpoint_path.exists():
            raise RuntimeError(f"No cached .pt2 ({pt2_path.name}) and no checkpoint ({checkpoint_path.name})")

        if not pt2_path.exists():
            model = load_model(checkpoint_path, upscale_factor)
            model.half()
            _log(f"Compiling torch_tensorrt .pt2 ({input_size[0]}x{input_size[1]}) -> {pt2_path.name} ...")
            t0 = time.perf_counter()
            export_trt(_IO.wrap(model), pt2_path, (input_size[0], input_size[1]))
            _log(f".pt2 compile done in {time.perf_counter() - t0:.1f}s")
        else:
            _log(f"Reusing cached .pt2 model {pt2_path.name}")

        self.model = torch.export.load(str(pt2_path)).module().cuda()
        self._io = _PinnedIO()
        _log("PT2Backend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # The compiled module expects fp16 input directly (the raw runner casts on-GPU instead).
        return self._io.download(self.model(self._io.upload(frame).half()))


class ONNXBackend:
    """onnxruntime backend of the bgr -> bgr VideoWrapper ONNX."""

    PROVIDER_MAP = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": [("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CUDAExecutionProvider"],
        "directml": ["DmlExecutionProvider"],
        "openvino": [("OpenVINOExecutionProvider", {"device_type": "GPU", "precision": "FP16"})],
        "cpu": ["CPUExecutionProvider"],
    }

    def __init__(self, model, input_size, upscale_factor: int, provider: str = "cuda"):
        import onnxruntime as ort
        checkpoint_path, tag = cache_paths.resolve_model(model, input_size, upscale_factor)
        print(ort.get_available_providers())

        onnx_path = get_onnx_video(checkpoint_path, cache_paths.onnx_video(tag, _IO.tag),
                                   input_size, upscale_factor, _IO)

        _log(f"Creating onnxruntime session (provider={provider}) ...")
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(str(onnx_path), providers=self.PROVIDER_MAP[provider])
        _log(f"onnxruntime session ready in {time.perf_counter() - t0:.1f}s")
        _log("ONNXBackend ready.")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"input": frame})[0]


class NCNNBackend:
    """ncnn-Vulkan backend (model + *255/clamp, no VideoWrapper; NCNNRunner does the BGR pre/post-processing).
    Takes and returns BGR frames."""

    def __init__(self, model, input_size, upscale_factor: int):
        from .export_ncnn import export_onnx_ncnn, export_ncnn, NCNNRunner
        checkpoint_path, tag = cache_paths.resolve_model(model, input_size, upscale_factor)

        param_path, bin_path = cache_paths.ncnn_paths(tag)
        param_path.parent.mkdir(parents=True, exist_ok=True)

        # ncnn files first, otherwise convert from the ncnn-source .onnx, exporting that from the checkpoint
        if not param_path.exists():
            onnx_path = cache_paths.onnx(tag)
            if not onnx_path.exists():
                if not checkpoint_path.exists():
                    raise RuntimeError(f"No cached ONNX ({onnx_path.name}) and no checkpoint ({checkpoint_path.name})")
                onnx_path.parent.mkdir(parents=True, exist_ok=True)
                _log(f"Exporting ncnn-source ONNX ({input_size[0]}x{input_size[1]}) -> {onnx_path.name} ...")
                export_onnx_ncnn(load_model(checkpoint_path, upscale_factor), onnx_path, input_size)
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


def make_backend(runtype: Runtype, model, input_size, upscale_factor: int):
    """Build the cv2-display backend for a runtype; "onnxruntime-<provider>" picks the ORT provider."""
    if runtype == "tensorrt":
        _log("Setting up TensorRT backend")
        return TRTBackend(model, input_size, upscale_factor)
    if runtype == "tensorrt-pt2":
        # torch_tensorrt .pt2: safer for large models where the raw engine build OOMs (see PT2Backend).
        _log("Setting up torch_tensorrt (.pt2) backend")
        return PT2Backend(model, input_size, upscale_factor)
    if runtype == "ncnn-vulkan":
        _log("Preparing ncnn-Vulkan backend (conversion on first run can take a while) ...")
        return NCNNBackend(model, input_size, upscale_factor)
    if runtype.startswith("onnxruntime-"):
        provider = runtype.split("-", 1)[1]  # cuda / tensorrt / openvino / directml / cpu
        _log(f"Preparing onnxruntime backend (provider={provider}; export on first run can take a while) ...")
        return ONNXBackend(model, input_size, upscale_factor, provider=provider)
    raise ValueError(f"Unknown runtype: {runtype}")
