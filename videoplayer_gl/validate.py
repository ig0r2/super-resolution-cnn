"""
Numeric parity check: run the GLSL render graph and the PyTorch model on the same LR frame and
compare the HR outputs. Catches weight-layout, orientation, pixel-shuffle and border-padding bugs
without needing a visible window (uses a hidden GL context).

    python -m videoplayer_gl.validate                      # default checkpoint, scales 2/3/4
    python -m videoplayer_gl.validate --model multiscale/SR_FastEDSR_4_128 --scales 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.checkpoints import load_model_from_checkpoint       # noqa: E402
from utils.path import get_checkpoints_path                      # noqa: E402
from videoplayer_gl.shader_gen import build_passes              # noqa: E402
from videoplayer_gl.gl_runtime import GLUpscaler                 # noqa: E402


def reference(model, frame_chw_uint8: torch.Tensor, scale: int) -> np.ndarray:
    """The exact computation VideoWrapperNVDEC bakes in, in fp32, as a (H,W,3) uint8 array."""
    with torch.no_grad():
        x = (frame_chw_uint8.float() / 255.0).unsqueeze(0)
        out = model(x, upscale_factor=scale)
        out = torch.clamp(out, 0.0, 1.0) * 255.0
        out = out.squeeze(0).round().to(torch.uint8)  # (3,Hr,Wr)
    return out.permute(1, 2, 0).cpu().numpy()


def run_scale(model, state_np, num_blocks, nf, scale, lr_h, lr_w, chunk_size) -> bool:
    torch.manual_seed(0)
    frame = torch.randint(0, 256, (3, lr_h, lr_w), dtype=torch.uint8)

    ref = reference(model, frame, scale)

    passes = build_passes(state_np, num_blocks, nf, scale, chunk_size=chunk_size)
    eng = GLUpscaler(passes, lr_h, lr_w, scale, win_w=8, win_h=8, visible=False)
    eng.infer(frame.cuda())
    got = eng.read_final()
    eng.close()

    d = np.abs(ref.astype(np.int16) - got.astype(np.int16))
    # Ignore a 1px frame: border pixels blend the exact-vs-emulated padding differently.
    inner = d[1:-1, 1:-1]
    max_d, mean_d = int(inner.max()), float(inner.mean())
    p999 = int(np.percentile(inner, 99.9))
    ok = max_d <= 4 and mean_d < 0.5
    print(f"  scale {scale}x  {lr_w}x{lr_h}->{lr_w*scale}x{lr_h*scale}  passes={len(passes):4d}  "
          f"max_d={max_d:3d}  p99.9_d={p999:3d}  mean_d={mean_d:.4f}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="multiscale/SR_FastEDSR_4_32")
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--size", type=int, nargs=2, default=[48, 64], metavar=("H", "W"))
    ap.add_argument("--chunk-size", type=int, default=8)
    args = ap.parse_args()

    ckpt = get_checkpoints_path(f"{args.model}.pth")
    model, cfg = load_model_from_checkpoint(ckpt, "cpu")
    model.eval()
    num_blocks, nf = cfg["params"]["num_blocks"], cfg["params"]["nf"]
    state_np = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    print(f"{args.model}  (num_blocks={num_blocks}, nf={nf})")

    lr_h, lr_w = args.size
    all_ok = all(run_scale(model, state_np, num_blocks, nf, s, lr_h, lr_w, args.chunk_size)
                 for s in args.scales)
    print("ALL OK" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
