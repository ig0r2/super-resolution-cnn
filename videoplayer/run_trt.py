import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from utils.path import get_project_root, get_checkpoints_path
from videoplayer.backends import TRTBackend
from videoplayer.player import VideoPlayer
from videoplayer.scaling import choose_auto_scale, get_screen_size

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/ldv-001-480p.mp4")
CANDIDATE_SCALES = (2, 3, 4)


################################################

def log(msg):
    print(f"[videoplayer] {msg}")


checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")
if not checkpoint_path.exists():
    print(f"Checkpoint for {MODEL} doesnt exist")
    sys.exit(1)

log(f"Opening video {VIDEO_PATH.name}")
probe_cap = cv2.VideoCapture(str(VIDEO_PATH))
frame_size = (int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
probe_cap.release()

screen_size = get_screen_size()
decision = choose_auto_scale(frame_size, screen_size, CANDIDATE_SCALES)
log(f"Frame {frame_size} | Screen {screen_size} | Model scale {decision.model_scale}x "
    f"-> {decision.model_output_size} | Target (bicubic) -> {decision.target_size}")

checkpoint_name = MODEL.split("/", 1)[-1]
tag = f"{checkpoint_name}_{frame_size[0]}x{frame_size[1]}_{decision.model_scale}x_cv2"
cache_dir = get_project_root("exports/videoplayer_trt")

log("Setting up TensorRT backend")
backend = TRTBackend(checkpoint_path, cache_dir, tag, frame_size, decision.model_scale)

target_size = decision.target_size if decision.target_size != decision.model_output_size else None

log("Opening player window")
player = VideoPlayer(VIDEO_PATH, target_size=target_size)

log("Starting playback.")
player.set_upscale_fn(backend).play()
