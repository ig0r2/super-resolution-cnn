import sys
from pathlib import Path

# Run as a script: put the project root on sys.path so `videoplayer_nvdec` / `utils` import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.path import get_project_root, get_checkpoints_path  # noqa: E402
from videoplayer import cache_paths  # noqa: E402
from videoplayer_nvdec.backend import TRTBackendNVDEC  # noqa: E402
from videoplayer_nvdec.player import NvdecVideoPlayer  # noqa: E402

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/ldv-001-480p.mp4")
CANDIDATE_SCALES = (2, 3, 4)

################################################


def log(msg):
    print(f"[videoplayer_nvdec] {msg}")


player = NvdecVideoPlayer(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
frame_size = player.size

checkpoint_name = MODEL.split("/", 1)[-1]
tag = f"{checkpoint_name}_{frame_size[0]}x{frame_size[1]}_{scale}x"

# The checkpoint is only needed to build the engine; if it's already cached, don't require it.
checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")
if not cache_paths.engine_nvdec(tag).exists() and not checkpoint_path.exists():
    print(f"No cached engine and no checkpoint for {MODEL}")
    sys.exit(1)

log("Setting up TensorRT backend (NVDEC decode)")
backend = TRTBackendNVDEC(checkpoint_path, tag, frame_size, scale)

log("Starting playback.")
player.set_upscale_fn(backend).play()
