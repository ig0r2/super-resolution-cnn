import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.path import get_project_root
from videoplayer.decode import Decoder
from videoplayer.players.factory import Display, Runtime, make_player

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/ldv-001-480p.mp4")
CANDIDATE_SCALES = (2, 3, 4)

# Supported combinations (anything else prints this table and exits):
# +----------------------------------------------------+------------------+---------+---------------------+
# | runtime                                            | decoder          | display | player              |
# +====================================================+==================+=========+=====================+
# | tensorrt, tensorrt-pt2, ncnn-vulkan, onnxruntime-* | pyav, cv2        | cv2     | VideoPlayerCV2      |
# | tensorrt                                           | nvdec            | cv2     | VideoPlayerNvdecCV2 |
# | tensorrt                                           | nvdec            | opengl  | VideoPlayerNvdecGL  |
# | opengl                                             | nvdec, pyav, cv2 | opengl  | VideoPlayerGL       |
# +----------------------------------------------------+------------------+---------+---------------------+
#
# Runtime:
#    "tensorrt", "tensorrt-pt2",
#    "ncnn-vulkan",
#    "onnxruntime-{cuda,tensorrt,openvino,directml,cpu}"
#    "opengl"
RUNTIME: Runtime = "tensorrt"
# Decoder: "pyav" | "cv2" | "nvdec"
DECODER: Decoder = "nvdec"
# Display: "cv2" | "opengl"
DISPLAY: Display = "cv2"

################################################


try:
    player = make_player(VIDEO_PATH, MODEL, RUNTIME, DECODER, DISPLAY, candidate_scales=CANDIDATE_SCALES)
except (RuntimeError, ValueError) as e:
    print(e)
    sys.exit(1)

player.play()
