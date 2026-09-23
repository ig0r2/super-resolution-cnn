import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.path import get_project_root, get_checkpoints_path
from videoplayer.backends import NCNNBackend, ONNXBackend, PT2Backend, TRTBackend
from videoplayer.player import VideoPlayer

BackendType = Literal["ncnn", "onnx", "trt", "pt2"]
ProviderType = Literal["cuda", "tensorrt", "directml", "openvino", "cpu"]

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/ldv-001-480p.mp4")
CANDIDATE_SCALES = (2, 3, 4)

BACKEND: BackendType = "trt"
PROVIDER: ProviderType = "directml"  # onnx only


################################################

def log(msg):
    print(f"[videoplayer] {msg}")


player = VideoPlayer(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
frame_size = player.size

checkpoint_name = MODEL.split("/", 1)[-1]
tag = f"{checkpoint_name}_{frame_size[0]}x{frame_size[1]}_{scale}x"

checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")

try:
    if BACKEND == "ncnn":
        log("Preparing ncnn-Vulkan backend (conversion on first run can take a while) ...")
        backend = NCNNBackend(checkpoint_path, tag, frame_size, scale)
    elif BACKEND == "onnx":
        log(f"Preparing onnxruntime backend (provider={PROVIDER}; export on first run can take a while) ...")
        backend = ONNXBackend(checkpoint_path, tag, frame_size, scale, provider=PROVIDER)
    elif BACKEND == "trt":
        log("Setting up TensorRT backend")
        backend = TRTBackend(checkpoint_path, tag, frame_size, scale)
    elif BACKEND == "pt2":
        log("Setting up torch_tensorrt (.pt2) backend")
        backend = PT2Backend(checkpoint_path, tag, frame_size, scale)
    else:
        raise ValueError(f"Unknown BACKEND: {BACKEND}")
except RuntimeError as e:
    print(e)
    sys.exit(1)

log("Starting playback.")
player.set_upscale_fn(backend).play()
