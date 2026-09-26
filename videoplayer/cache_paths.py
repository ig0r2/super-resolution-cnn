"""Single source of truth for where exported SR artifacts are cached.

Artifacts are keyed by `tag` (model_size_scale) plus, for the video-wrapped exports, the wrapper's
I/O (`VideoIO.tag`: "rgb_bgr", "rgb_rgb_f16", "bgr_bgr" or "rgb_rgb"), and grouped by kind, NOT by which script
built them, so the run players and the perf evaluators share the same files:

    exports/onnx_video/{tag}__rgb_bgr.onnx    rgb -> bgr wrapper ONNX (NVDEC + cv2 display)
    exports/trt_video/{tag}__rgb_bgr.engine   its TensorRT engine
    exports/pt2_video/{tag}__rgb_bgr.pt2      its torch_tensorrt module (large-model fallback)
    exports/onnx_video/{tag}__rgb_rgb_f16.onnx rgb -> fp16 NCHW RGB (NVDEC, downscaled display)
    exports/trt_video/{tag}__rgb_rgb_f16.engine its TensorRT engine
    exports/onnx_cv2/{tag}.onnx               bgr -> bgr wrapper (PyAV / cv2 decode + cv2 display)
    exports/trt_cv2/{tag}.engine              ... its TensorRT engine
    exports/pt2_cv2/{tag}.pt2                 ... its torch_tensorrt module
    exports/onnx_nvdec/{tag}.onnx             rgb -> rgb wrapper (NVDEC + CUDA-GL display)
    exports/trt_nvdec/{tag}.engine            ... its TensorRT engine
    exports/onnx/{tag}.onnx                   CHW RGB model + *255/clamp (ncnn/pnnx source)
    exports/ncnn/{tag}.ncnn.param/bin         ncnn model

bgr -> bgr and rgb -> rgb keep the directories they had before the VideoIO scheme, so engines
built earlier are still reused.
"""
from pathlib import Path

from utils.path import get_checkpoints_path, get_project_root

def resolve_model(model, frame_size, scale):
    """model: checkpoint name under checkpoints/ ("multiscale/SR_FastEDSR_4_128") or a .pth path.
    Returns (checkpoint_path, tag), where tag = "{stem}_{h}x{w}_{s}x" keys all cached exports."""
    path = Path(model)
    if path.suffix != ".pth":
        path = get_checkpoints_path(f"{model}.pth")
    return path, f"{path.stem}_{frame_size[0]}x{frame_size[1]}_{scale}x"


# VideoIO.tag -> legacy directory suffix (exports/{kind}_{suffix}/{tag}.{ext})
_LEGACY = {"bgr_bgr": "cv2", "rgb_rgb": "nvdec"}


def _video(kind, ext, tag, io):
    if io in _LEGACY:
        return get_project_root(f"exports/{kind}_{_LEGACY[io]}/{tag}.{ext}")
    return get_project_root(f"exports/{kind}_video/{tag}__{io}.{ext}")


def onnx_video(tag, io):
    return _video("onnx", "onnx", tag, io)


def engine_video(tag, io):
    return _video("trt", "engine", tag, io)


def pt2_video(tag, io):
    return _video("pt2", "pt2", tag, io)


def onnx(tag):
    return get_project_root(f"exports/onnx/{tag}.onnx")


def ncnn_paths(tag):
    d = get_project_root("exports/ncnn")
    return d / f"{tag}.ncnn.param", d / f"{tag}.ncnn.bin"
