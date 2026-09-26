import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.path import get_project_root
from videoplayer.backends.gl_build import build_engine
from videoplayer.cache_paths import resolve_model
from videoplayer.players.gl import VideoPlayerGL

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/frantic.mp4")
CANDIDATE_SCALES = (2, 3, 4)
CHUNK_SIZE = 8  # input groups per conv chunk (must keep bound textures <= GL_MAX_TEXTURE_IMAGE_UNITS)

################################################


player = VideoPlayerGL(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
lr_h, lr_w = player.size

# Window opens at the display target if we have one, else the native SR output size.
win_h, win_w = player.target_size if player.target_size is not None else (lr_h * scale, lr_w * scale)

checkpoint_path, _ = resolve_model(MODEL, (lr_h, lr_w), scale)
print(f"[videoplayer] Loading checkpoint {MODEL} and compiling GLSL shader graph ...")
engine, meta = build_engine(checkpoint_path, scale, lr_h, lr_w,
                            win_w=win_w, win_h=win_h, title="SR Video (GL)",
                            fullscreen=player.fullscreen, chunk_size=CHUNK_SIZE)
print(f"[videoplayer] {meta['num_passes']} render passes (num_blocks={meta['num_blocks']}, nf={meta['nf']}, {scale}x)")

print("[videoplayer] Starting playback (pure OpenGL, no ML runtime).")
player.set_engine(engine).play()
