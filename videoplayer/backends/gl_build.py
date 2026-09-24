"""
Factory that turns a checkpoint into a ready GLUpscaler. Shared by the player and the evaluator.

Only SR_FastEDSR_Multi is supported (the architecture the shader generator understands); any other
model raises RuntimeError so callers can skip it the same way they skip a failed engine build.
"""

from pathlib import Path
from typing import Tuple

from utils.checkpoints import load_model_from_checkpoint
from .gl_engine import GLUpscaler
from .shader_gen import build_passes

SUPPORTED = "SR_FastEDSR_Multi"


def build_engine(checkpoint_path, scale: int, lr_h: int, lr_w: int,
                 win_w: int, win_h: int, title: str = "SR Video (GL)",
                 fullscreen: bool = False, visible: bool = True,
                 chunk_size: int = 8) -> Tuple[GLUpscaler, dict]:
    model, cfg = load_model_from_checkpoint(Path(checkpoint_path), "cpu")
    if cfg["name"] != SUPPORTED:
        raise RuntimeError(f"GL backend only supports {SUPPORTED}, got {cfg['name']}")

    num_blocks, nf = cfg["params"]["num_blocks"], cfg["params"]["nf"]
    state_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    passes = build_passes(state_np, num_blocks, nf, scale, chunk_size=chunk_size)

    engine = GLUpscaler(passes, lr_h, lr_w, scale, win_w=win_w, win_h=win_h,
                        title=title, fullscreen=fullscreen, visible=visible)
    meta = {"num_blocks": num_blocks, "nf": nf, "num_passes": len(passes)}
    return engine, meta
