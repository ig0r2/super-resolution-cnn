import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.path import get_project_root, get_checkpoints_path
from videoplayer.backends.gl_build import build_engine
from videoplayer.players.gl import VideoPlayerGL

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/frantic.mp4")
CANDIDATE_SCALES = (2, 3, 4)
CHUNK_SIZE = 8  # input groups per conv chunk (must keep bound textures <= GL_MAX_TEXTURE_IMAGE_UNITS)


################################################


def log(msg):
    print(f"[videoplayer] {msg}")


player = VideoPlayerGL(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
lr_h, lr_w = player.size

# Window opens at the display target if we have one, else the native SR output size.
win_h, win_w = player.target_size if player.target_size is not None else (lr_h * scale, lr_w * scale)

log(f"Loading checkpoint {MODEL} and compiling GLSL shader graph ...")
engine, meta = build_engine(get_checkpoints_path(f"{MODEL}.pth"), scale, lr_h, lr_w,
                            win_w=win_w, win_h=win_h, title="SR Video (GL)",
                            fullscreen=player.fullscreen, chunk_size=CHUNK_SIZE)
log(f"{meta['num_passes']} render passes (num_blocks={meta['num_blocks']}, nf={meta['nf']}, {scale}x)")

log("Starting playback (pure OpenGL, no ML runtime).")
player.set_engine(engine).play()
