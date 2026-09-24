import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.path import get_project_root, get_checkpoints_path
from videoplayer.backends.nvdec_backend import TRTBackendNVDEC
from videoplayer.players.nvdec_gl import VideoPlayerNvdecGL

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/ldv-001-480p.mp4")
CANDIDATE_SCALES = (2, 3, 4)


################################################


def log(msg):
    print(f"[videoplayer] {msg}")


player = VideoPlayerNvdecGL(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
frame_size = player.size

checkpoint_name = MODEL.split("/", 1)[-1]
tag = f"{checkpoint_name}_{frame_size[0]}x{frame_size[1]}_{scale}x"

checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")

log("Setting up TensorRT backend (NVDEC decode, CUDA-GL display)")
try:
    backend = TRTBackendNVDEC(checkpoint_path, tag, frame_size, scale)
except RuntimeError as e:
    print(e)
    sys.exit(1)

log("Starting playback (CUDA-GL zero-copy display).")
player.set_upscale_fn(backend).play()
