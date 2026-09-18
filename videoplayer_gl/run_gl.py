import sys
from pathlib import Path

# Run as a script: put the project root on sys.path so `videoplayer_gl` / `utils` import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.checkpoints import load_model_from_checkpoint  # noqa: E402
from utils.path import get_project_root, get_checkpoints_path  # noqa: E402
from videoplayer_gl.gl_runtime import GLUpscaler  # noqa: E402
from videoplayer_gl.player_gl import GLVideoPlayer  # noqa: E402
from videoplayer_gl.shader_gen import build_passes  # noqa: E402

MODEL = "multiscale/SR_FastEDSR_4_128"
VIDEO_PATH = get_project_root("videoinput/frantic.mp4")
CANDIDATE_SCALES = (2, 3, 4)
CHUNK_SIZE = 8  # input groups per conv chunk (must keep bound textures <= GL_MAX_TEXTURE_IMAGE_UNITS)

################################################


def log(msg):
    print(f"[videoplayer_gl] {msg}")


player = GLVideoPlayer(VIDEO_PATH)
scale = player.configure_scale(CANDIDATE_SCALES)
lr_h, lr_w = player.size

log(f"Loading checkpoint {MODEL} and compiling GLSL shader graph ...")
model, cfg = load_model_from_checkpoint(get_checkpoints_path(f"{MODEL}.pth"), "cpu")
num_blocks, nf = cfg["params"]["num_blocks"], cfg["params"]["nf"]
state_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

passes = build_passes(state_np, num_blocks, nf, scale, chunk_size=CHUNK_SIZE)
log(f"{len(passes)} render passes (num_blocks={num_blocks}, nf={nf}, {scale}x)")

# Window opens at the display target if we have one, else the native SR output size.
win_h, win_w = player.target_size if player.target_size is not None else (lr_h * scale, lr_w * scale)
engine = GLUpscaler(passes, lr_h, lr_w, scale, win_w=win_w, win_h=win_h,
                    title="SR Video (GL)", fullscreen=player.fullscreen)

log("Starting playback (pure OpenGL, no ML runtime).")
player.set_engine(engine).play()
