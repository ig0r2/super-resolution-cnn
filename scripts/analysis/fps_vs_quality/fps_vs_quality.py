"""
fps_vs_quality.py - grafik odnosa brzine i kvaliteta sa Pareto frontom.

Spaja dva CSV-a po imenu modela: merenje brzine (FPS) i merenje kvaliteta
(SSIM ili LPIPS), pa crta rasejani grafik brzina (log osa) naspram kvaliteta,
sa istaknutim Pareto frontom - skupom konfiguracija koje nijedna druga ne
nadmasuje istovremeno i po brzini i po kvalitetu. Vertikalne isprekidane linije
oznacavaju pragove realnog vremena (30 i 60 FPS).

Ovakav prikaz je pravi izbor za pitanje "koji model za realno vreme": tacke
ispod/levo su losije od fronta, a front pokazuje najbolji dostizni kvalitet za
svaku brzinu. Za metriku LPIPS nize je bolje (osa se invertuje).

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Sve konfiguracije se obradjuju u jednom pokretanju i daju odvojenu sliku
(imenovanu po "name" polju) u results/analysis/fps_vs_quality/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.plotting import (ordered_archs, pareto_front, save_fig, set_fps_ticks,
                                     setup_style, style_for)
import matplotlib.pyplot as plt

from utils.analysis.config import run_configs
from utils.analysis.data import derive_label, name_filter, read_rows, require_results, to_float
from utils.analysis.metrics import check_metrics, higher_is_better
from utils.analysis.names import INTERPOLATIONS, arch_of, is_multiscale
from utils.path import get_results_path
from utils.plot import dot_to_comma

# Interpolacione metode (nearest, bilinear...) su u istom CSV-u kao modeli, pa se i one izbacuju.
DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg", *INTERPOLATIONS]

PARETO_MODES = ("global", "arch", "both", None)

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "default",              # koristi se za ime izlazne slike
    "fps_results": "results_2x_DIV2K_half.csv",  # CSV sa merenjem brzine
    "quality_results": "results_2x_DIV2K_half.csv",  # CSV sa merenjem kvaliteta
    "fps_col": "FPS 720p",          # kolona sa FPS vrednoscu (npr. FPS 720p, FPS 480p)
    "metric": "SSIM",               # SSIM/PSNR: vise je bolje, LPIPS: nize je bolje
    "out": None,                    # None -> ..._<name>.png
    "archs": None,                  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,                # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,                # None -> DEFAULT_EXCLUDE
    "min_params": None,
    "max_params": None,
    "include_multiscale": True,     # False -> izbaci multiscale modele (bez \d+x tokena u imenu)
    "thresholds": [30, 60],         # pragovi FPS-a (vertikalne linije); [] za bez
    "label": None,                  # None -> izvedi iz imena fajlova
    "pareto": "global",             # "global" (jedan front preko svih modela), "arch" (front
                                    # po arhitekturi, ostale tacke blede), "both" ili None
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "SSIM_720p",
        "metric": "SSIM",
        "fps_col": "FPS 720p",
    },
    {
        "name": "SSIM_480p",
        "metric": "SSIM",
        "fps_col": "FPS 480p",
    },
    {
        "name": "LPIPS_720p",
        "metric": "LPIPS",
        "fps_col": "FPS 720p",
    },
    {
        "name": "LPIPS_480p",
        "metric": "LPIPS",
        "fps_col": "FPS 480p",
    },
    {
        "name": "SSIM_720p_pareto_arch",
        "metric": "SSIM",
        "fps_col": "FPS 720p",
        "pareto": "arch",
    },
    {
        "name": "SSIM_480p_pareto_arch",
        "metric": "SSIM",
        "fps_col": "FPS 480p",
        "pareto": "arch",
    },
    {
        "name": "LPIPS_720p_pareto_arch",
        "metric": "LPIPS",
        "fps_col": "FPS 720p",
        "pareto": "arch",
    },
    {
        "name": "LPIPS_480p_pareto_arch",
        "metric": "LPIPS",
        "fps_col": "FPS 480p",
        "pareto": "arch",
    },
]


##############################################


def read_quality(path: Path, metric: str) -> dict:
    """Mapa model_name -> vrednost metrike."""
    out = {}
    for r in read_rows(path, [metric]):
        v = to_float(r.get(metric))
        if r.get("model_name") and v is not None:
            out[r["model_name"]] = v
    return out


def load_rows(fps_path, quality_path, cfg) -> list[dict]:
    """Spoji brzinu i kvalitet po imenu modela i primeni filtere."""
    keep = name_filter(DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude, cfg.archs, cfg.include)
    quality = read_quality(quality_path, cfg.metric)

    rows = []
    for r in read_rows(fps_path, [cfg.fps_col]):
        name = r.get("model_name", "")
        if name not in quality or not keep(name):
            continue
        if not cfg.include_multiscale and is_multiscale(name):
            continue
        params = to_float(r.get("params"))
        fps = to_float(r.get(cfg.fps_col))
        if not fps:
            continue
        if params and cfg.min_params and params < cfg.min_params:
            continue
        if params and cfg.max_params and params > cfg.max_params:
            continue
        rows.append({"name": name, "arch": arch_of(name), "fps": fps, "q": quality[name]})
    return rows


def front_of(rows, maximize):
    """Nedominirane tacke: veci FPS i bolji kvalitet."""
    return pareto_front(rows, x_higher=True, y_higher=maximize, x="fps", y="q")


def plot(rows, archs, cfg, label, maximize):
    colors, markers = style_for(archs)
    setup_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    if cfg.pareto not in PARETO_MODES:
        sys.exit(f"Nepoznat pareto rezim '{cfg.pareto}'. Dozvoljeno: {PARETO_MODES}")
    per_arch = cfg.pareto in ("arch", "both")

    for a in archs:
        sub = [r for r in rows if r["arch"] == a]
        if not sub:
            continue
        xs = [r["fps"] for r in sub]
        ys = [r["q"] for r in sub]
        if not per_arch:
            ax.scatter(xs, ys, c=colors[a], marker=markers[a], s=42, alpha=0.85,
                       label=a, edgecolors="white", linewidths=0.4)
            continue
        ax.scatter(xs, ys, c=colors[a], marker=markers[a], s=25, alpha=0.18,
                   edgecolors="none")
        front = front_of(sub, maximize)
        fx = [r["fps"] for r in front]
        fy = [r["q"] for r in front]
        ax.plot(fx, fy, color=colors[a], linewidth=1.6, alpha=0.9)
        ax.scatter(fx, fy, c=colors[a], marker=markers[a], s=42, alpha=0.95,
                   label=a, edgecolors="white", linewidths=0.5, zorder=3)

    if cfg.pareto in ("global", "both"):
        front = front_of(rows, maximize)
        ax.plot([r["fps"] for r in front], [r["q"] for r in front],
                "-", color="#333", lw=1.3, zorder=1, label="Pareto front")

    for thr in cfg.thresholds:
        ax.axvline(thr, color="#aaa", ls="--", lw=0.9, zorder=0)
        ax.text(thr, min(r["q"] for r in rows), f" {dot_to_comma(f'{thr:g}')} FPS", rotation=90,
                color="#888", fontsize=8.5, va="bottom", ha="left")

    ax.set_xscale("log")
    suffix = f" ({label})" if label else ""
    better = "vise je bolje" if maximize else "nize je bolje"
    ax.set_xlabel(f"Brzina inferencije - {cfg.fps_col} (FPS, log)")
    ax.set_ylabel(f"{cfg.metric}{suffix} - {better}")
    ax.set_title(f"Odnos brzine i kvaliteta ({cfg.metric})"
                 + (" - Pareto front po arhitekturi" if per_arch else ""))
    if not maximize:
        ax.invert_yaxis()

    lo, hi = min(r["fps"] for r in rows), max(r["fps"] for r in rows)
    set_fps_ticks(ax, lo * 0.9, hi * 1.1)

    loc = "lower left" if maximize else "upper right"
    ax.legend(loc=loc, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def run_config(cfg):
    check_metrics([cfg.metric])
    fps_path = require_results(cfg.fps_results)
    quality_path = require_results(cfg.quality_results)

    rows = load_rows(fps_path, quality_path, cfg)
    if not rows:
        sys.exit("Nijedan model nema i brzinu i kvalitet uz zadate filtere.")

    archs = ordered_archs(r["arch"] for r in rows)
    label = cfg.label if cfg.label is not None else derive_label(cfg.quality_results, cfg.fps_results)
    maximize = higher_is_better(cfg.metric)

    fig = plot(rows, archs, cfg, label, maximize)
    out_path = save_fig(fig, get_results_path("analysis/fps_vs_quality")
                        / (cfg.out or f"fps_vs_quality_{cfg.name}.png"))

    counts = {a: sum(1 for r in rows if r["arch"] == a) for a in archs}
    front = front_of(rows, maximize)
    print(f"Brzina:  {fps_path}  (kolona {cfg.fps_col})")
    print(f"Kvalitet:{quality_path}  (metrika {cfg.metric})")
    print(f"Modela:  {len(rows)}  ({', '.join(f'{a}:{n}' for a, n in counts.items())})")
    print(f"Pareto front ({len(front)}): {', '.join(r['name'] for r in front)}")
    print(f"Slika:   {out_path}")


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
