"""
Exportuje sve checkpoint-e iz checkpoints/ (osim *_latest) u slim .pth pod exports/pth/,
sa istom strukturom foldera. Slim checkpoint ima samo model_config i model_state_dict,
sto je dovoljno za videoplayer, evaluaciju, inferencu i export u ONNX/ncnn/TRT/GLSL.

Postojeci slim checkpoint se preskace ako nije stariji od izvornog.
"""

from utils.checkpoints import ensure_slim
from utils.path import get_checkpoints_path, get_slim_checkpoints_path

if __name__ == "__main__":
    ckpt_root = get_checkpoints_path()
    total_src = total_dst = 0
    for src in sorted(ckpt_root.rglob("*.pth")):
        if src.stem.endswith("_latest"):
            continue
        rel = src.relative_to(ckpt_root)
        dst = ensure_slim(src, get_slim_checkpoints_path(rel))
        total_src += src.stat().st_size
        total_dst += dst.stat().st_size
        print(f"{rel.with_suffix('').as_posix():55s} {src.stat().st_size / 1024:10.1f} KB -> {dst.stat().st_size / 1024:10.1f} KB")

    print(f"\nUkupno: {total_src / 2**20:.1f} MB -> {total_dst / 2**20:.1f} MB")
