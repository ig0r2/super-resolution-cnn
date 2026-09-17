import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.path import get_project_root, get_checkpoints_path
from videoplayer.backends import NCNNBackend
from videoplayer.player import VideoPlayer

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

checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")

log("Preparing ncnn-Vulkan backend (conversion on first run can take a while) ...")
try:
    backend = NCNNBackend(checkpoint_path, tag, frame_size, scale)
except RuntimeError as e:
    print(e)
    sys.exit(1)

log("Starting playback.")
player.set_upscale_fn(backend).play()
