"""
multiscale.py - multi-scale naspram namenskih (single-scale) modela.

Multi-scale model je treniran zajednicki za vise faktora uvecanja i u imenu nema
oznaku faktora (npr. `RFDN_4_256`); namenski (single-scale) model ima oznaku
(npr. `RFDN_2x_2_256`). Ova skripta uparuje multi-scale i single-scale model iste
konfiguracije (ista arhitektura, broj blokova i kanala) na istom faktoru i crta
njihov kvalitet jedan naspram drugog, sa linijom identiteta (y = x).

Tacke iznad dijagonale znace da je multi-scale model bolji od namenskog za tu
metriku (za LPIPS, gde je nize bolje, obe ose su obrnute pa vazi isto). Time se vidi da li
deljenje parametara izmedju faktora kosta kvalitet.

Moze se zadati vise faktora odjednom preko "scales"; svaki dobija svoj panel
jedan pored drugog, iz istog skupa (sablon imena "template"). Alternativno,
"results" zadaje tacno jedan CSV (jedan panel), a "scales" se ostavi None.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.plotting import ordered_archs, save_fig, setup_style, style_for
import matplotlib.pyplot as plt

from utils.analysis.config import run_configs
from utils.analysis.data import name_filter, parse_results_name, read_rows, require_results, to_float
from utils.analysis.fmt import zf
from utils.analysis.metrics import check_metrics, higher_is_better
from utils.analysis.names import INTERPOLATIONS, arch_of, pair_multiscale
from utils.path import get_results_path

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "default",              # koristi se za ime izlazne slike
    "scales": None,                 # lista faktora (svaki svoj panel), npr. [2, 3, 4];
                                    # None -> koristi se "results" (jedan panel)
    "dataset": "Set14",             # skup za sve faktore kad se koristi "scales"
    "template": "results_{scale}x_{dataset}_half.csv",  # sablon CSV-a po faktoru
    "results": "results_2x_DIV2K_half.csv",  # jedan CSV (kad je "scales" None)
    "metric": "SSIM",               # SSIM/PSNR: vise je bolje, LPIPS: nize je bolje
    "out": None,                    # None -> multiscale_<name>.png
    "archs": None,                  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,                # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,                # None -> DEFAULT_EXCLUDE
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "SSIM_DIV2K",
        "scales": [2, 3, 4],
        "dataset": "DIV2K",
        "metric": "SSIM",
    },
    {
        "name": "LPIPS_DIV2K",
        "scales": [2, 3, 4],
        "dataset": "DIV2K",
        "metric": "LPIPS",
    },
]


##############################################


def compute_pairs(path: Path, metric, keep):
    """Vrati listu uparenih {arch, single, multi} za jedan CSV."""
    items = []
    for r in read_rows(path, [metric]):
        name = r.get("model_name", "")
        if name in INTERPOLATIONS or not keep(name):
            continue
        v = to_float(r.get(metric))
        if v is not None:
            items.append((name, v))
    return [{"arch": arch_of(m_name), "single": s_val, "multi": m_val}
            for _, (_, s_val), (m_name, m_val) in pair_multiscale(items)]


def run_config(cfg):
    check_metrics([cfg.metric])
    keep = name_filter(DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude, cfg.archs, cfg.include)

    # Izvori: ili vise faktora ("scales") ili jedan CSV ("results").
    sources = []  # (naslov_faktora, dataset, path)
    if cfg.scales:
        for s in cfg.scales:
            path = require_results(cfg.template.format(scale=s, dataset=cfg.dataset))
            sources.append((f"$\\times${s}", cfg.dataset, path))
    else:
        path = require_results(cfg.results)
        dataset, factor = parse_results_name(cfg.results)
        sources.append((f"$\\times${factor}" if factor else "", dataset, path))

    maximize = higher_is_better(cfg.metric)

    panels = []  # (naslov, dataset, pairs)
    for title, dataset, path in sources:
        pairs = compute_pairs(path, cfg.metric, keep)
        if not pairs:
            print(f"Upozorenje: nema uparenih konfiguracija u {path.name}.")
            continue
        panels.append((title, dataset, pairs))
    if not panels:
        sys.exit("Nema uparenih multi-scale / single-scale konfiguracija.")

    archs = ordered_archs(p["arch"] for _, _, pairs in panels for p in pairs)
    colors, markers = style_for(archs)

    setup_style(cfg.dpi)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5.4), squeeze=False)
    axes = axes[0]

    stats = []
    for ax, (title, dataset, pairs) in zip(axes, panels):
        for a in archs:
            xs = [p["single"] for p in pairs if p["arch"] == a]
            ys = [p["multi"] for p in pairs if p["arch"] == a]
            if xs:
                ax.scatter(xs, ys, c=colors[a], marker=markers[a], s=46, alpha=0.85,
                           label=a, edgecolors="white", linewidths=0.4)
        allv = [p["single"] for p in pairs] + [p["multi"] for p in pairs]
        lo, hi = min(allv), max(allv)
        pad = (hi - lo) * 0.05 or 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#888", lw=1,
                label="y = x")
        ds = f"{dataset}, " if dataset else ""
        ax.set_xlabel(f"{cfg.metric} - namenski ({ds}{title})")
        ax.set_ylabel(f"{cfg.metric} - multi-scale ({ds}{title})")
        deltas = [(p["multi"] - p["single"]) * (1 if maximize else -1) for p in pairs]
        wins = sum(1 for d in deltas if d > 0)
        if not maximize:
            # LPIPS: nize je bolje - obe ose obrnute, pa i ovde "iznad dijagonale" = multi bolji.
            ax.invert_xaxis()
            ax.invert_yaxis()
        ax.legend(loc="best", fontsize=9)
        stats.append((title, len(pairs), wins, sum(deltas) / len(deltas)))

    fig.tight_layout()
    out_path = save_fig(fig, get_results_path("analysis/multiscale") / (cfg.out or f"multiscale_{cfg.name}.png"))

    print(f"Metrika: {cfg.metric}  ({'vise je bolje' if maximize else 'nize je bolje'})")
    for title, n_pairs, wins, mean_d in stats:
        clean = title.replace("$\\times$", "x")
        print(f"  {clean}: {n_pairs} parova, multi bolji u {wins}, "
              f"prosecna razlika (u korist boljeg) {zf(mean_d, '+.4f')}")
    print(f"Slika: {out_path}")


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
