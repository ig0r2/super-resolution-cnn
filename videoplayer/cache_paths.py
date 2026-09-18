"""Single source of truth for where exported SR artifacts are cached.

Artifacts are keyed only by `tag` (model_size_scale) and grouped by kind, NOT by which script
built them, so the run players and the perf evaluators share the same files instead of each
rebuilding into its own folder:

    exports/onnx_cv2/{tag}.onnx        HWC BGR wrapper  (TRT engine build + onnxruntime)
    exports/onnx/{tag}.onnx            CHW RGB bare model (ncnn/pnnx source; reusable by any bare-model backend)
    exports/onnx_nvdec/{tag}.onnx      CHW RGB wrapper  (NVDEC TRT engine build)
    exports/trt_cv2/{tag}.engine       cv2-decode TensorRT engine
    exports/trt_nvdec/{tag}.engine     NVDEC-decode TensorRT engine
    exports/pt2_cv2/{tag}.pt2          cv2-decode torch_tensorrt module (large-model fallback)
    exports/ncnn/{tag}.ncnn.param/bin  ncnn model (no ONNX; pnnx from the bare model)
"""
from utils.path import get_project_root


def onnx_cv2(tag):
    return get_project_root(f"exports/onnx_cv2/{tag}.onnx")


def engine_cv2(tag):
    return get_project_root(f"exports/trt_cv2/{tag}.engine")


def pt2_cv2(tag):
    return get_project_root(f"exports/pt2_cv2/{tag}.pt2")


def onnx(tag):
    return get_project_root(f"exports/onnx/{tag}.onnx")


def ncnn_paths(tag):
    d = get_project_root("exports/ncnn")
    return d / f"{tag}.ncnn.param", d / f"{tag}.ncnn.bin"


def onnx_nvdec(tag):
    return get_project_root(f"exports/onnx_nvdec/{tag}.onnx")


def engine_nvdec(tag):
    return get_project_root(f"exports/trt_nvdec/{tag}.engine")
