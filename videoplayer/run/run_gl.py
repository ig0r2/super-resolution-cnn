import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.path import get_project_root
from videoplayer.players.player_gl import VideoPlayerGL

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/frantic.mp4")
CANDIDATE_SCALES = (2, 3, 4)
CHUNK_SIZE = 8  # input groups per conv chunk (must keep bound textures <= GL_MAX_TEXTURE_IMAGE_UNITS)

################################################


player = VideoPlayerGL(VIDEO_PATH, MODEL, candidate_scales=CANDIDATE_SCALES, chunk_size=CHUNK_SIZE)
print("[videoplayer] Starting playback (pure OpenGL, no ML runtime).")
player.play()
