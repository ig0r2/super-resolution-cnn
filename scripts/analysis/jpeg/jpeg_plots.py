"""
jpeg_plots.py - korist JPEG-treniranih modela na kompresovanom ulazu.

Na ulazu koji je JPEG-kompresovan, poredi kvalitet standardnih modela i modela
treniranih sa JPEG degradacijom (ime sadrzi `jpeg`), u zavisnosti od broja
parametara. Dve grupe se boje razlicito, a opciono se crta vodoravna linija bazne
interpolacije (podrazumevano bicubic, iskljucena po defaultu). Ocekivani nalaz:
JPEG-trenirani modeli su iznad standardnih, a deo standardnih modela pada i ispod
bazne interpolacije, jer izostravaju blok-artefakte umesto da ih potiskuju.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.plotting import save_fig, set_param_ticks, setup_style
import matplotlib.pyplot as plt

from utils.analysis.config import run_configs
from utils.analysis.data import derive_label, name_filter, read_rows, require_results, to_float
from utils.analysis.fmt import zf
from utils.analysis.metrics import check_metrics, higher_is_better
from utils.analysis.names import INTERPOLATIONS
from utils.path import get_results_path

DEFAULT_EXCLUDE = ["GAN", "ESRGAN"]
STD_COLOR, JPEG_COLOR = "#1b6ca8", "#e8702a"

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "SSIM",  # koristi se za ime izlazne slike
    "results": "results_2x_DIV2K_jpeg_half.csv",  # CSV evaluiran na JPEG ulazu
    "metric": "SSIM",  # SSIM/PSNR: vise je bolje, LPIPS: nize je bolje
    "show_baseline": False,  # True -> nacrtaj vodoravnu liniju bazne interpolacije
    "baseline": "bicubic",  # koja bazna interpolacija (kad je show_baseline True)
    "out": None,  # None -> jpeg_plots_<name>.png
    "archs": None,  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,  # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,  # None -> DEFAULT_EXCLUDE
    "min_params": None,
    "max_params": None,
    "label": None,  # None -> izvedi iz imena fajla
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "SSIM_DIV2K",
        "metric": "SSIM",
    },
    {
        "name": "LPIPS_DIV2K",
        "metric": "LPIPS",
    },
    {
        "name": "SSIM_Set14",
        "results": "results_2x_Set14_jpeg_half.csv",
        "metric": "SSIM",
    },
    {
        "name": "LPIPS_Set14",
        "results": "results_2x_Set14_jpeg_half.csv",
        "metric": "LPIPS",
    },
]


##############################################


def run_config(cfg):
    check_metrics([cfg.metric])
    keep = name_filter(DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude, cfg.archs, cfg.include)
    draw_baseline = cfg.show_baseline and cfg.baseline and cfg.baseline != "none"
    path = require_results(cfg.results)

    std, jpeg = [], []
    baseline_val = None
    for r in read_rows(path, [cfg.metric]):
        name = r.get("model_name", "")
        v = to_float(r.get(cfg.metric))
        if not name or v is None:
            continue
        if draw_baseline and name == cfg.baseline:
            baseline_val = v
        if name in INTERPOLATIONS or not keep(name):
            continue
        params = to_float(r.get("params"))
        if not params:
            continue
        if cfg.min_params and params < cfg.min_params:
            continue
        if cfg.max_params and params > cfg.max_params:
            continue
        (jpeg if "jpeg" in name else std).append((params, v))

    if not std and not jpeg:
        sys.exit("Nema modela za zadate filtere.")
    if draw_baseline and baseline_val is None:
        print(f"Upozorenje: bazni red '{cfg.baseline}' nije nadjen u CSV-u.")

    maximize = higher_is_better(cfg.metric)
    label = cfg.label if cfg.label is not None else derive_label(cfg.results)

    setup_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    for pts, color, marker, lab in (
            (std, STD_COLOR, "o", "standardni modeli"),
            (jpeg, JPEG_COLOR, "s", "JPEG-trenirani modeli")):
        if pts:
            ax.scatter([p for p, _ in pts], [v for _, v in pts], c=color, marker=marker,
                       s=44, alpha=0.85, label=lab, edgecolors="white", linewidths=0.4)

    if baseline_val is not None:
        ax.axhline(baseline_val, color="#c0392b", ls="--", lw=1.1)
        ax.text(0.99, baseline_val, f" {cfg.baseline} (bazna linija)",
                transform=ax.get_yaxis_transform(), color="#c0392b",
                fontsize=8.5, ha="right", va="bottom")

    ax.set_xscale("log")
    set_param_ticks(ax, [p for p, _ in std + jpeg])

    if not maximize:
        ax.invert_yaxis()
    suffix = f" ({label})" if label else ""
    better = "vise je bolje" if maximize else "nize je bolje"
    ax.set_xlabel("Broj parametara (log)")
    ax.set_ylabel(f"{cfg.metric}{suffix} - {better}")
    ax.set_title(f"Kvalitet na JPEG-kompresovanom ulazu ({cfg.metric})")
    ax.legend(loc="lower right" if maximize else "upper right", fontsize=9)
    fig.tight_layout()

    out_path = save_fig(fig, get_results_path("analysis/jpeg") / (cfg.out or f"jpeg_plots_{cfg.name}.png"))

    print(f"Izvor: {path}  (metrika {cfg.metric})")
    print(f"Standardnih: {len(std)}, JPEG-treniranih: {len(jpeg)}")
    if baseline_val is not None:
        cmp = (lambda v: v < baseline_val) if maximize else (lambda v: v > baseline_val)
        below = sum(1 for _, v in std if cmp(v))
        print(f"Bazna linija ({cfg.baseline}) {cfg.metric} = {zf(baseline_val, '.4f')}; "
              f"standardnih ispod baze: {below}")
    print(f"Slika: {out_path}")


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
