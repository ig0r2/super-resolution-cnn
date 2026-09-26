import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.path import get_project_root
from videoplayer.backends.nvdec_backend import TRTBackendNVDEC
from videoplayer.players.nvdec import VideoPlayerNvdecCV2

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/ldv-001-480p.mp4")
CANDIDATE_SCALES = (2, 3, 4)

################################################


player = VideoPlayerNvdecCV2(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
frame_size = player.size

print("[videoplayer] Setting up TensorRT backend (NVDEC decode)")
try:
    backend = TRTBackendNVDEC(MODEL, frame_size, scale, output=player.sr_output())
except RuntimeError as e:
    print(e)
    sys.exit(1)

print("[videoplayer] Starting playback.")
player.set_upscale_fn(backend).play()
