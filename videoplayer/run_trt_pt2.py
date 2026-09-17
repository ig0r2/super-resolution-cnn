import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.path import get_project_root, get_checkpoints_path
from videoplayer import cache_paths
from videoplayer.backends import PT2Backend
from videoplayer.player import VideoPlayer

# torch_tensorrt (.pt2) player. Prefer run_trt.py (raw TensorRT engine) for small/medium models --
# it's a bit faster. Use THIS for LARGE models: building a raw engine can OOM (TensorRT needs a big
# build-time workspace + tactic-profiling VRAM on top of the weights), whereas the torch_tensorrt
# .pt2 compile builds more conservatively and still runs where the engine build fails.

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/ldv-001-480p.mp4")
CANDIDATE_SCALES = (2, 3, 4)


################################################

def log(msg):
    print(f"[videoplayer] {msg}")


player = VideoPlayer(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
frame_size = player.size

checkpoint_name = MODEL.split("/", 1)[-1]
tag = f"{checkpoint_name}_{frame_size[0]}x{frame_size[1]}_{scale}x"

# No ONNX fallback here (torch_tensorrt compiles the model directly), so a cache miss needs the .pth.
checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")
if not cache_paths.pt2_cv2(tag).exists() and not checkpoint_path.exists():
    print(f"No cached .pt2 and no checkpoint for {MODEL}")
    sys.exit(1)

log("Setting up torch_tensorrt (.pt2) backend")
backend = PT2Backend(checkpoint_path, tag, frame_size, scale)

log("Starting playback.")
player.set_upscale_fn(backend).play()
