import copy

import torch
import yaml

from utils.trainers.trainer import Trainer
from utils.trainers.trainer_gan import TrainerGAN
from utils.logger import Logger
from utils.path import get_config_path, get_logs_path


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into a copy of `base`.
    Dict values are merged; all other types are replaced outright.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_model(entry: dict, defaults: dict) -> dict:
    """
    Merge fallbacks -> defaults -> model entry, then flatten into the
    dict shape that BaseTrainer.set_model() expects.
    """
    merged = _deep_merge(defaults, entry)

    # Pull out model-identity keys before flattening.
    model_cfg = {
        'name': merged.pop('name', None),
        'checkpoint_name': merged.pop('checkpoint_name'),
        'pretrain_checkpoint_path': merged.pop('pretrain_checkpoint', None),
        'params': merged.pop('params', {}),
    }

    # Flatten training + data into the top level (what BaseTrainer reads).
    flat = {}
    flat.update(merged.pop('training', {}))
    flat.update(merged.pop('data', {}))
    flat.update(merged)  # mode, multiscale, jpeg_degradation, any extras
    flat['model'] = model_cfg

    return flat


def load_config(path) -> list[dict]:
    """
    Load config and for each model in models, combine model settings and default settings
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get('defaults', {})
    models = raw.get('models', [])

    return [_resolve_model(entry, defaults) for entry in models]


def trainer_key(cfg: dict) -> tuple:
    """
    The subset of settings that determine which trainer class and which
    DataLoaders are built. If this tuple is identical to the previous
    run's, the trainer can be reused.
    """
    return (cfg['mode'], cfg['multiscale'], cfg['jpeg_degradation'],
            cfg['patch_size'], cfg['batch_size'], cfg['num_workers'],
            cfg['train_preload'], cfg['val_preload'],
            cfg['model']['params'].get('upscale_factor'))


if __name__ == "__main__":
    CONFIG_FILE = "training_ESRGAN.yaml"

    ######################################################
    models_config = load_config(get_config_path(CONFIG_FILE))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.backends.cudnn.is_available(): torch.backends.cudnn.benchmark = True

    trainer = None
    current_key = None

    for cfg in models_config:
        key = trainer_key(cfg)

        if key != current_key:
            TrainerClass = TrainerGAN if cfg['mode'] == 'gan' else Trainer
            trainer = TrainerClass(config=cfg, device=device)
            current_key = key

        with Logger(get_logs_path(f"training/training_{cfg['model']['checkpoint_name']}.txt")):
            trainer.set_model(cfg).train()
