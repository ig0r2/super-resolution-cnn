"""
multiscale.py - multi-scale naspram namenskih (single-scale) modela.

Multi-scale model je treniran zajednicki za vise faktora uvecanja i u imenu nema
oznaku faktora (npr. `RFDN_4_256`); namenski (single-scale) model ima oznaku
(npr. `RFDN_2x_2_256`). Ova skripta uparuje multi-scale i single-scale model iste
konfiguracije (ista arhitektura, broj blokova i kanala) na istom faktoru i crta
njihov kvalitet jedan naspram drugog, sa linijom identiteta (y = x).

Tacke iznad dijagonale znace da je multi-scale model bolji od namenskog za tu
metriku (za LPIPS je "bolje" nize, pa se osa tumaci obrnuto). Time se vidi da li
deljenje parametara izmedju faktora kosta kvalitet.

Moze se zadati vise faktora odjednom preko "scales"; svaki dobija svoj panel
jedan pored drugog, iz istog skupa (sablon imena "template"). Alternativno,
"results" zadaje tacno jedan CSV (jedan panel), a "scales" se ostavi None.
"""

import csv
import sys
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path import get_results_path

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]
METRICS = ("SSIM", "PSNR", "LPIPS")
# Bazne interpolacije nisu modeli - izbacuju se iz poredjenja.
BASELINE_NAMES = ("nearest", "bilinear", "bicubic", "lanczos")

PALETTE = {
    "SRCNN": "#7f7f7f", "VDSR": "#9467bd", "SRResNet": "#17becf",
    "EDSR": "#1b6ca8", "FastEDSR": "#e8702a", "IMDN": "#3c9a5f", "RFDN": "#b13b8f",
}
MARKERS = {
    "SRCNN": "P", "VDSR": "X", "SRResNet": "*",
    "EDSR": "o", "FastEDSR": "s", "IMDN": "^", "RFDN": "D",
}
PREFERRED_ORDER = ["SRCNN", "VDSR", "SRResNet", "EDSR", "FastEDSR", "IMDN", "RFDN"]
KNOWN_DATASETS = ("Set5", "Set14", "BSD100", "Urban100", "DIV2K")

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


def is_single_scale(name: str) -> bool:
    return any(t.endswith("x") and t[:-1].isdigit() for t in name.split("_"))


def config_key(name: str) -> str:
    """Ime bez `SR_` prefiksa i bez tokena faktora - kljuc za uparivanje."""
    parts = [t for t in name.split("_") if t != "SR"
             and not (t.endswith("x") and t[:-1].isdigit())]
    return "_".join(parts)


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def label_from_filename(name):
    parts = Path(name).stem.split("_")
    factor = next((p[:-1] for p in parts if p.endswith("x") and p[:-1].isdigit()), None)
    dataset = next((p for p in parts if p in KNOWN_DATASETS), None)
    return dataset, factor


def compute_pairs(path: Path, metric, exclude, archs_filter, include):
    """Vrati listu uparenih {arch, single, multi} za jedan CSV."""
    multi, single = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if metric not in reader.fieldnames:
            sys.exit(f"Metrika '{metric}' ne postoji u {path.name}.")
        for r in reader:
            name = r.get("model_name", "")
            if not name or name in BASELINE_NAMES:
                continue
            if any(sub in name for sub in exclude):
                continue
            if include and not any(sub in name for sub in include):
                continue
            if archs_filter and arch_of(name) not in archs_filter:
                continue
            v = to_float(r.get(metric))
            if v is None:
                continue
            (single if is_single_scale(name) else multi)[config_key(name)] = (name, v)

    pairs = []
    for key in set(multi) & set(single):
        m_name, m_val = multi[key]
        s_name, s_val = single[key]
        pairs.append({"arch": arch_of(m_name), "single": s_val, "multi": m_val})
    return pairs


def style_for(archs):
    extra_c = cycle(["#d62728", "#bcbd22", "#8c564b", "#e377c2", "#2ca02c"])
    extra_m = cycle(["v", "<", ">", "p", "h"])
    colors = {a: PALETTE.get(a) or next(extra_c) for a in archs}
    markers = {a: MARKERS.get(a) or next(extra_m) for a in archs}
    return colors, markers


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

    # Izvori: ili vise faktora ("scales") ili jedan CSV ("results").
    sources = []  # (naslov_faktora, dataset, path)
    if cfg.scales:
        for s in cfg.scales:
            path = resolve_results(cfg.template.format(scale=s, dataset=cfg.dataset))
            if not path.exists():
                sys.exit(f"CSV za faktor x{s} ne postoji: {path}")
            sources.append((f"$\\times${s}", cfg.dataset, path))
    else:
        path = resolve_results(cfg.results)
        if not path.exists():
            sys.exit(f"CSV ne postoji: {path}")
        dataset, factor = label_from_filename(cfg.results)
        sources.append((f"$\\times${factor}" if factor else "", dataset, path))

    maximize = cfg.metric != "LPIPS"

    panels = []  # (naslov, dataset, pairs)
    for title, dataset, path in sources:
        pairs = compute_pairs(path, cfg.metric, exclude, archs_filter, include)
        if not pairs:
            print(f"Upozorenje: nema uparenih konfiguracija u {path.name}.")
            continue
        panels.append((title, dataset, pairs))
    if not panels:
        sys.exit("Nema uparenih multi-scale / single-scale konfiguracija.")

    present = set()
    for _, _, pairs in panels:
        present |= {p["arch"] for p in pairs}
    archs = [a for a in PREFERRED_ORDER if a in present] + sorted(present - set(PREFERRED_ORDER))
    colors, markers = style_for(archs)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": cfg.dpi})
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
        # ax.set_title(f"{title} - {len(pairs)} parova (multi bolji {wins})")
        ax.legend(loc="best", fontsize=9)
        stats.append((title, len(pairs), wins, sum(deltas) / len(deltas)))

    fig.tight_layout()
    out_name = cfg.out or f"multiscale_{cfg.name}.png"
    out_path = get_results_path("analysis/multiscale") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Metrika: {cfg.metric}  ({'vise je bolje' if maximize else 'nize je bolje'})")
    for title, n_pairs, wins, mean_d in stats:
        clean = title.replace("$\\times$", "x")
        print(f"  {clean}: {n_pairs} parova, multi bolji u {wins}, "
              f"prosecna razlika (u korist boljeg) {zf(mean_d, '+.4f')}")
    print(f"Slika: {out_path}")


if __name__ == "__main__":
    main()
