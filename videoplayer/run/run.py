import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.path import get_project_root
from videoplayer.backends.cv2_backends import Runtype
from videoplayer.players.player_cv2 import VideoPlayerCV2

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/frantic.mp4")
CANDIDATE_SCALES = (2, 3, 4)

BACKEND: Runtype = "onnxruntime-directml"
# CPU decoder: "pyav" (FFmpeg via PyAV) or "cv2" (cv2.VideoCapture); both feed the same BGR engine
DECODER: Literal["pyav", "cv2"] = "pyav"

################################################


try:
    player = VideoPlayerCV2(VIDEO_PATH, MODEL, runtype=BACKEND, decoder=DECODER,
                            candidate_scales=CANDIDATE_SCALES)
except (RuntimeError, ValueError) as e:
    print(e)
    sys.exit(1)

player.play()
