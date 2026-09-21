"""
jpeg_plots.py - korist JPEG-treniranih modela na kompresovanom ulazu.

Na ulazu koji je JPEG-kompresovan, poredi kvalitet standardnih modela i modela
treniranih sa JPEG degradacijom (ime sadrzi `jpeg`), u zavisnosti od broja
parametara. Dve grupe se boje razlicito, a opciono se crta vodoravna linija bazne
interpolacije (podrazumevano bicubic, iskljucena po defaultu). Ocekivani nalaz:
JPEG-trenirani modeli su iznad standardnih, a deo standardnih modela pada i ispod
bazne interpolacije, jer izostravaju blok-artefakte umesto da ih potiskuju.
"""

import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path import get_results_path

DEFAULT_EXCLUDE = ["GAN", "ESRGAN"]
METRICS = ("SSIM", "PSNR", "LPIPS")
BASELINE_NAMES = ("nearest", "bilinear", "bicubic", "lanczos")

STD_COLOR, JPEG_COLOR = "#1b6ca8", "#e8702a"
PARAM_TICKS = [1e3, 1e4, 1e5, 1e6, 1e7]
KNOWN_DATASETS = ("Set5", "Set14", "BSD100", "Urban100", "DIV2K")

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


def resolve_results(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return get_results_path(name)


def zf(value, spec) -> str:
    """Formatira broj po `spec` sa decimalnim zarezom umesto tacke."""
    return format(value, spec).replace(".", ",")


def arch_of(name: str) -> str:
    parts = name.split("_")
    if parts and parts[0] == "SR":
        parts = parts[1:]
    return parts[0] if parts else name


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def derive_label(results_name, override):
    if override is not None:
        return override
    parts = Path(results_name).stem.split("_")
    factor = next((p[:-1] for p in parts if p.endswith("x") and p[:-1].isdigit()), None)
    dataset = next((p for p in parts if p in KNOWN_DATASETS), None)
    return ", ".join(b for b in (dataset, f"$\\times${factor}" if factor else None) if b)


def main():
    if not CONFIGS:
        sys.exit("CONFIGS je prazna - dodaj bar jednu konfiguraciju.")
    for i, cfg_dict in enumerate(CONFIGS):
        cfg = SimpleNamespace(**{**CONFIG_DEFAULTS, **cfg_dict})
        if i:
            print("\n" + "=" * 60 + "\n")
        print(f"### Konfiguracija: {cfg.name}")
        run_config(cfg)


def run_config(cfg):
    if cfg.metric not in METRICS:
        sys.exit(f"Nepoznata metrika '{cfg.metric}'. Podrzane: {', '.join(METRICS)}")

    exclude = DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude
    archs_filter = set(cfg.archs) if cfg.archs else None
    include = cfg.include or None
    draw_baseline = cfg.show_baseline and cfg.baseline and cfg.baseline != "none"

    path = resolve_results(cfg.results)
    if not path.exists():
        sys.exit(f"CSV ne postoji: {path}")

    std, jpeg = [], []
    baseline_val = None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if cfg.metric not in reader.fieldnames:
            sys.exit(f"Metrika '{cfg.metric}' ne postoji u {path.name}.")
        for r in reader:
            name = r.get("model_name", "")
            if not name:
                continue
            v = to_float(r.get(cfg.metric))
            if v is None:
                continue
            if draw_baseline and name == cfg.baseline:
                baseline_val = v
            if name in BASELINE_NAMES:
                continue
            if any(sub in name for sub in exclude):
                continue
            if include and not any(sub in name for sub in include):
                continue
            if archs_filter and arch_of(name) not in archs_filter:
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

    maximize = cfg.metric != "LPIPS"
    label = derive_label(cfg.results, cfg.label)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": cfg.dpi})
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
    all_p = [p for p, _ in std + jpeg]
    xt = [t for t in PARAM_TICKS if min(all_p) * 0.9 <= t <= max(all_p) * 1.1]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"$10^{{{int(round(math.log10(t)))}}}$" for t in xt])

    if not maximize:
        ax.invert_yaxis()
    suffix = f" ({label})" if label else ""
    better = "vise je bolje" if maximize else "nize je bolje"
    ax.set_xlabel("Broj parametara (log)")
    ax.set_ylabel(f"{cfg.metric}{suffix} - {better}")
    ax.set_title(f"Kvalitet na JPEG-kompresovanom ulazu ({cfg.metric})")
    ax.legend(loc="lower right" if maximize else "upper right", fontsize=9)
    fig.tight_layout()

    out_name = cfg.out or f"jpeg_plots_{cfg.name}.png"
    out_path = get_results_path("analysis/jpeg") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Izvor: {path}  (metrika {cfg.metric})")
    print(f"Standardnih: {len(std)}, JPEG-treniranih: {len(jpeg)}")
    if baseline_val is not None:
        cmp = (lambda v: v < baseline_val) if maximize else (lambda v: v > baseline_val)
        below = sum(1 for _, v in std if cmp(v))
        print(f"Bazna linija ({cfg.baseline}) {cfg.metric} = {zf(baseline_val, '.4f')}; "
              f"standardnih ispod baze: {below}")
    print(f"Slika: {out_path}")


if __name__ == "__main__":
    main()
