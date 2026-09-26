import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.path import get_project_root
from videoplayer.backends.cv2_backends import Runtype, make_backend
from videoplayer.players.cv2_player import VideoPlayerCV2

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/frantic.mp4")
CANDIDATE_SCALES = (2, 3, 4)

BACKEND: Runtype = "onnxruntime-directml"
# CPU decoder: "pyav" (FFmpeg via PyAV) or "cv2" (cv2.VideoCapture); both feed the same BGR engine
DECODER: Literal["pyav", "cv2"] = "pyav"

################################################


player = VideoPlayerCV2(VIDEO_PATH, decoder=DECODER)
scale = player.configure_scale(CANDIDATE_SCALES)
frame_size = player.size
io = player.video_io()

print(f"[videoplayer] Wrapper I/O: {io.tag}")
try:
    backend = make_backend(BACKEND, MODEL, frame_size, scale, io)
except (RuntimeError, ValueError) as e:
    print(e)
    sys.exit(1)

print("[videoplayer] Starting playback.")
player.set_upscale_fn(backend).play()
