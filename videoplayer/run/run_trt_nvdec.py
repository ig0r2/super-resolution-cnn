import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.path import get_project_root
from videoplayer.backends.nvdec_backend import TRTBackendNVDEC

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/ldv-001-480p.mp4")
CANDIDATE_SCALES = (2, 3, 4)

# Display: "cv" (device->host copy + cv2.imshow) or "gl" (CUDA-GL zero-copy)
DISPLAY: Literal["cv", "gl"] = "gl"

################################################


if DISPLAY == "gl":
    from videoplayer.players.player_nvdec_gl import VideoPlayerNvdecGL as Player
else:
    from videoplayer.players.player_nvdec import VideoPlayerNvdecCV2 as Player

player = Player(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
frame_size = player.size

print(f"[videoplayer] Setting up TensorRT backend (NVDEC decode, {DISPLAY} display)")
try:
    backend = TRTBackendNVDEC(MODEL, frame_size, scale)
except RuntimeError as e:
    print(e)
    sys.exit(1)

print("[videoplayer] Starting playback.")
player.set_upscale_fn(backend).play()
