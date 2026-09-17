import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.path import get_project_root, get_checkpoints_path
from videoplayer import cache_paths
from videoplayer.backends import ONNXBackend
from videoplayer.player import VideoPlayer

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/ldv-001-480p.mp4")
CANDIDATE_SCALES = (2, 3, 4)
PROVIDER = "directml"  # one of: cuda, tensorrt, directml, openvino, cpu

################################################

def log(msg):
    print(f"[videoplayer] {msg}")


player = VideoPlayer(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
frame_size = player.size

checkpoint_name = MODEL.split("/", 1)[-1]
tag = f"{checkpoint_name}_{frame_size[0]}x{frame_size[1]}_{scale}x"

# The checkpoint is only needed to export the ONNX; if it's already cached, don't require it.
checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")
if not cache_paths.onnx_cv2(tag).exists() and not checkpoint_path.exists():
    print(f"No cached ONNX and no checkpoint for {MODEL}")
    sys.exit(1)

log(f"Preparing onnxruntime backend (provider={PROVIDER}; export on first run can take a while) ...")
backend = ONNXBackend(checkpoint_path, tag, frame_size, scale, provider=PROVIDER)

log("Starting playback.")
player.set_upscale_fn(backend).play()
