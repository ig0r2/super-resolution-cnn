from pathlib import Path

import torch
from models import get_model
from utils.path import get_checkpoints_path, get_slim_checkpoints_path


def load_model_from_checkpoint(path, device):
    """
    Ucitava checkpoint i instancira model
    :param path: putanja
    :param device: torch.device
    :return: (model, model_config iz checkpointa)
    """
    checkpoint = torch.load(path, map_location=device)
    model = get_model(checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint['model_config']


def load_model(model, device):
    """
    Kao load_model_from_checkpoint, ali uvek iz slim checkpointa (vidi resolve_checkpoint).
    :param model: ime pod checkpoints/ ("multiscale/SR_FastEDSR_4_128") ili putanja do .pth
    :param device: torch.device
    :return: (model, model_config iz checkpointa)
    """
    return load_model_from_checkpoint(resolve_checkpoint(model), device)


def save_slim_checkpoint(src, dst):
    """
    Cuva slim checkpoint: samo model_config i model_state_dict (bez optimizera, schedulera, istorije,
    diskriminatora...). load_model_from_checkpoint ga ucitava isto kao pun checkpoint.
    :param src: pun checkpoint iz treninga
    :param dst: putanja slim checkpointa
    :return: dst
    """
    checkpoint = torch.load(src, map_location="cpu")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_config': checkpoint['model_config'],
        'model_state_dict': checkpoint['model_state_dict'],
    }, dst)
    return dst


def ensure_slim(src, dst):
    """Pravi slim `dst` iz punog checkpointa `src` ako ne postoji ili je stariji od `src`."""
    src, dst = Path(src), Path(dst)
    if src.exists() and (not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime):
        print(f"Creating slim checkpoint {dst.name} from {src} ...")
        save_slim_checkpoint(src, dst)
    return dst


def resolve_checkpoint(model):
    """
    Vraca slim checkpoint pod exports/pth/ za `model`, praveci ga iz checkpoints/ pri prvoj upotrebi.
    Ako ne postoji ni slim ni pun checkpoint, vraca (nepostojecu) slim putanju.
    :param model: ime pod checkpoints/ ("multiscale/SR_FastEDSR_4_128") ili putanja do .pth
                  (pod checkpoints/ -> ista relativna putanja pod exports/pth/; pod exports/pth/ -> ona sama;
                  bilo gde drugde -> exports/pth/{stem}.pth)
    """
    path = Path(model)
    if path.suffix != ".pth":
        return ensure_slim(get_checkpoints_path(f"{model}.pth"), get_slim_checkpoints_path(f"{model}.pth"))

    path = path.resolve()
    if path.is_relative_to(get_slim_checkpoints_path()):
        return path
    ckpt_root = get_checkpoints_path()
    name = path.relative_to(ckpt_root) if path.is_relative_to(ckpt_root) else Path(path.name)
    return ensure_slim(path, get_slim_checkpoints_path(name))
