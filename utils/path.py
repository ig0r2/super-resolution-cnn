from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root(subpath=None) -> Path:
    if subpath: return PROJECT_ROOT / Path(subpath)
    return PROJECT_ROOT


def _get_custom_folder(folder, subpath=None) -> Path:
    path = get_project_root() / folder
    path.mkdir(parents=True, exist_ok=True)
    if subpath: path /= Path(subpath)
    return path


def get_logs_path(subpath=None) -> Path:
    return _get_custom_folder("logs", subpath)


def get_results_path(subpath=None) -> Path:
    return _get_custom_folder("results", subpath)


def get_checkpoints_path(subpath=None) -> Path:
    return _get_custom_folder("checkpoints", subpath)


def get_config_path(subpath=None) -> Path:
    return _get_custom_folder("config", subpath)


def get_data_path(subpath=None) -> Path:
    config_file = get_config_path("data.yaml")
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
        if "path" in config:
            path = Path(config["path"])
            path.mkdir(parents=True, exist_ok=True)
            if subpath: path /= Path(subpath)
            return path
    return _get_custom_folder("data", subpath)
