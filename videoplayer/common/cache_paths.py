"""Single source of truth for where exported SR artifacts are cached.

Artifacts live under exports/{kind}/{pipeline}/{tag}.{ext}: `kind` is the format, `pipeline` the
player path the export is wrapped for, and `tag` (model_size_scale, see resolve_model) the model.
They are grouped by pipeline, NOT by which script built them, so the run players and the perf
evaluators share the same files:

    exports/onnx/cv2/{tag}.onnx       bgr -> bgr VideoWrapper (PyAV / cv2 decode + cv2 display)
    exports/trt/cv2/{tag}.engine      ... its TensorRT engine
    exports/pt2/cv2/{tag}.pt2         ... its torch_tensorrt module (large-model fallback)
    exports/onnx/nvdec/{tag}.onnx     rgb -> fp16 NCHW RGB VideoWrapper (NVDEC, both displays)
    exports/trt/nvdec/{tag}.engine    ... its TensorRT engine
    exports/onnx/ncnn/{tag}.onnx      CHW RGB model + *255/clamp (ncnn/pnnx source)
    exports/ncnn/{tag}.ncnn.param/bin ncnn model
"""
from pathlib import Path
from typing import Literal

from utils.path import get_checkpoints_path, get_project_root

Pipeline = Literal["cv2", "nvdec", "ncnn"]


def resolve_model(model, frame_size, scale):
    """model: checkpoint name under checkpoints/ ("multiscale/SR_FastEDSR_4_128") or a .pth path.
    Returns (checkpoint_path, tag), where tag = "{stem}_{h}x{w}_{s}x" keys all cached exports."""
    path = Path(model)
    if path.suffix != ".pth":
        path = get_checkpoints_path(f"{model}.pth")
    return path, f"{path.stem}_{frame_size[0]}x{frame_size[1]}_{scale}x"


def _export(kind, pipeline: Pipeline, tag, ext):
    return get_project_root(f"exports/{kind}/{pipeline}/{tag}.{ext}")


def onnx(pipeline: Pipeline, tag):
    return _export("onnx", pipeline, tag, "onnx")


def engine(pipeline: Pipeline, tag):
    return _export("trt", pipeline, tag, "engine")


def pt2(pipeline: Pipeline, tag):
    return _export("pt2", pipeline, tag, "pt2")


def ncnn_paths(tag):
    d = get_project_root("exports/ncnn")
    return d / f"{tag}.ncnn.param", d / f"{tag}.ncnn.bin"
