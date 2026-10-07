"""
arch_panels.py - preglednija verzija grafika "kvalitet/brzina naspram velicine".

Kvalitet ili brzina naspram velicine modela, bez "spaghetti"
efekta kad je arhitektura mnogo. Dva rezima prikaza:

  - "panels": nekoliko panela sa po 3-4 arhitekture (npr. levo arhitekture iz
    literature, desno FastEDSR i ABPN uz EDSR kao referencu);
  - "multiples": po jedan mali panel za svaku arhitekturu (small multiples) -
    istaknuta arhitektura je u boji, a Pareto frontovi ostalih su sivi u pozadini,
    pa se poredjenje ne gubi.

Ose su proizvoljne kolone iz results CSV-a: x i y mogu biti "params", neka FPS
kolona ("FPS 480p") ili metrika (PSNR/SSIM/LPIPS). Pareto front se racuna po
smeru svake ose (manje parametara je bolje, vise FPS-a je bolje, nizi LPIPS je
bolji...).

Viseskalni modeli (bez Nx tokena u imenu) se podrazumevano izbacuju
(include_multiscale), kao u ostalim skriptama za poglavlje o arhitekturama.

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Slike se cuvaju u results/analysis/arch_panels/.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.plotting import FPS_TICKS, PALETTE, pareto_front, save_fig, setup_style
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from utils.analysis.config import run_configs
from utils.analysis.data import read_rows, require_results, to_float
from utils.analysis.fmt import zf
from utils.analysis.names import parse_name
from utils.path import get_results_path

OUT_DIR = "analysis/arch_panels"

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

# Namerno drugaciji markeri nego u ostalim graficima: u svakom panelu iz GROUPS
# serije dobijaju razlicite oblike (EDSR o, ostale P/X), boje ostaju iz PALETTE.
PANEL_MARKERS = {
    "SRCNN": "P", "VDSR": "X", "EDSR": "o", "FastEDSR": "P",
    "IMDN": "P", "RFDN": "X", "ABPN": "X",
}

# Kolona -> (oznaka ose, da li je vece bolje, logaritamska osa).
AXES = {
    "params": ("Broj parametara (log)", False, True),
    "PSNR": ("PSNR (dB)", True, False),
    "SSIM": ("SSIM", True, False),
    "LPIPS": ("LPIPS", False, False),
}

CONFIG_DEFAULTS = {
    "name": "default",  # ime izlazne slike
    "results": "results_2x_DIV2K_half.csv",
    "x": "params",  # kolona za x osu
    "y": "SSIM",  # kolona za y osu
    "mode": "panels",  # "panels" ili "multiples"
    # panels: lista {"title", "series"}; multiples: "series" je redosled malih panela
    "panels": None,
    "series": None,
    "ncols": 3,  # multiples: broj kolona mreze
    # Smer x ose za Pareto front; None -> iz AXES (manje parametara je bolje).
    # Za FPS naspram velicine stavi True: front je tada "najbrzi model za datu
    # velicinu" (vise parametara je dozvoljeno, pa se trazi gornja anvelopa).
    "x_higher": None,
    "exclude": None,  # None -> DEFAULT_EXCLUDE
    "include_multiscale": False,
    "show_dominated": False,  # blede tacke van fronta istaknute serije
    "hlines": [],  # horizontalne linije (npr. [30, 60] za FPS)
    "vlines": [],  # vertikalne linije
    "label": "DIV2K, ×2",  # dodaje se oznaci ose metrike
    "figsize": None,
    "font_size": 10,
    "marker_size": 40,  # velicina tacaka na frontu (panels)
    "legend_loc": None,  # None -> automatski; npr. "lower right"
    "dpi": 140,
}

##############################################

# Podela po prostoru i nacinu izracunavanja; EDSR je referenca u svakoj grupi
# (LR prostor, obicne konvolucije, bez reziduala nad uvecanim ulazom).
GROUPS = [
    {"title": "HR prostor", "series": ["EDSR", "VDSR"]},
    {"title": "LR: destilacija i pažnja", "series": ["EDSR", "IMDN", "RFDN"]},
    {"title": "LR: rezidual nad uvećanim ulazom", "series": ["EDSR", "FastEDSR", "ABPN"]},
]
ALL = ["VDSR", "EDSR", "IMDN", "RFDN", "ABPN", "FastEDSR"]  # SRCNN je jedna tacka - bez svog panela
GROUP_STYLE = {"figsize": (9.0, 3.4), "font_size": 8.5, "marker_size": 20}  # tri panela u sirini teksta

CONFIGS = []
for _y in ("SSIM", "LPIPS", "PSNR"):
    CONFIGS += [
        {"name": f"{_y}_vs_params_panels", "x": "params", "y": _y, "panels": GROUPS, **GROUP_STYLE,
         "legend_loc": "lower right"},
        {"name": f"{_y}_vs_params_multiples", "x": "params", "y": _y, "mode": "multiples", "series": ALL},
    ]
CONFIGS += [
    {"name": "FPS_vs_params_panels", "x": "params", "y": "FPS 480p", "panels": GROUPS,
     "hlines": [], "x_higher": True, **GROUP_STYLE},
    {"name": "FPS_vs_params_multiples", "x": "params", "y": "FPS 480p", "mode": "multiples",
     "series": ALL, "hlines": [], "x_higher": True},
]


##############################################


def axis_info(col):
    if col in AXES:
        return AXES[col]
    if col.startswith("FPS"):
        return f"Brzina inferencije - {col} (log)", True, True
    sys.exit(f"Nepoznata kolona za osu: '{col}'")


def load_rows(cfg):
    path = require_results(cfg.results)
    exclude = DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude
    rows = []
    for r in read_rows(path, [cfg.x, cfg.y]):
        name = r.get("model_name", "")
        if not name.startswith("SR_") or any(s in name for s in exclude):
            continue
        series, config, multiscale = parse_name(name)
        if multiscale and not cfg.include_multiscale:
            continue
        x, y = to_float(r.get(cfg.x)), to_float(r.get(cfg.y))
        if not x or y is None:
            continue
        rows.append({"name": name, "series": series, "config": config, "x": x, "y": y})
    return path, rows


def style_axes(ax, cfg, xlim, ylim, show_xlabel=True, show_ylabel=True):
    xlabel, _, xlog = axis_info(cfg.x)
    ylabel, y_higher, ylog = axis_info(cfg.y)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if not y_higher and not ylog:
        ax.invert_yaxis()
    for col, axis in ((cfg.x, ax.xaxis), (cfg.y, ax.yaxis)):
        if col.startswith("FPS"):
            lo, hi = (xlim if axis is ax.xaxis else ylim)
            ticks = [t for t in FPS_TICKS if min(lo, hi) <= t <= max(lo, hi)]
            (ax.set_xticks if axis is ax.xaxis else ax.set_yticks)(ticks)
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(plt.NullFormatter())
    if show_xlabel:
        ax.set_xlabel(xlabel)
    if show_ylabel:
        lab = ylabel if cfg.y.startswith("FPS") else f"{ylabel} ({cfg.label})"
        ax.set_ylabel(lab)
    for h in cfg.hlines:
        ax.axhline(h, color="#aaa", ls="--", lw=0.9, zorder=0)
        ax.text(xlim[1], h, f"{zf(h, 'g')} FPS ", color="#888", fontsize=7.5, va="bottom", ha="right")
    for v in cfg.vlines:
        ax.axvline(v, color="#aaa", ls="--", lw=0.9, zorder=0)


def limits(rows, col):
    vals = [r["x" if col == "x" else "y"] for r in rows]
    return min(vals), max(vals)


def padded(lo, hi, log):
    if log:
        return lo / 1.4, hi * 1.4
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def draw_series(ax, sub, s, x_higher, y_higher, show_dominated, size=40):
    color, marker = PALETTE.get(s, "#555"), PANEL_MARKERS.get(s, "o")
    front = pareto_front(sub, x_higher, y_higher)
    if show_dominated:
        rest = [r for r in sub if r not in front]
        ax.scatter([r["x"] for r in rest], [r["y"] for r in rest], c=color, marker=marker,
                   s=size * 0.5, alpha=0.2, edgecolors="none")
    ax.plot([r["x"] for r in front], [r["y"] for r in front], color=color, lw=1.6, alpha=0.9)
    ax.scatter([r["x"] for r in front], [r["y"] for r in front], c=color, marker=marker, s=size,
               edgecolors="white", linewidths=0.5, zorder=3, label=s)


def run_config(cfg):
    path, rows = load_rows(cfg)
    if not rows:
        sys.exit("Nema modela posle filtera.")
    _, x_higher, xlog = axis_info(cfg.x)
    _, y_higher, ylog = axis_info(cfg.y)
    if cfg.x_higher is not None:
        x_higher = cfg.x_higher
    xlim = padded(*limits(rows, "x"), xlog)
    ylim = padded(*limits(rows, "y"), ylog)
    setup_style(cfg.dpi, cfg.font_size)

    if cfg.mode == "panels":
        panels = cfg.panels or [{"title": None, "series": sorted({r["series"] for r in rows})}]
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=cfg.figsize or (5.4 * n, 4.6), sharey=True, squeeze=False)
        for i, (ax, p) in enumerate(zip(axes[0], panels)):
            for s in p["series"]:
                sub = [r for r in rows if r["series"] == s]
                if sub:
                    draw_series(ax, sub, s, x_higher, y_higher, cfg.show_dominated, size=cfg.marker_size)
            style_axes(ax, cfg, xlim, ylim, show_ylabel=(i == 0))
            if p.get("title"):
                ax.set_title(p["title"], fontsize=cfg.font_size + 1)
            loc = ("lower right" if y_higher else "upper right") if cfg.x == "params" and not cfg.y.startswith("FPS") \
                else "upper right"
            ax.legend(loc=cfg.legend_loc or loc, fontsize=cfg.font_size - 1.5, framealpha=0.9)
    elif cfg.mode == "multiples":
        series = [s for s in (cfg.series or sorted({r["series"] for r in rows}))
                  if any(r["series"] == s for r in rows)]
        ncols = min(cfg.ncols, len(series))
        nrows = math.ceil(len(series) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=cfg.figsize or (3.3 * ncols, 2.9 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        fronts = {s: pareto_front([r for r in rows if r["series"] == s], x_higher, y_higher) for s in series}
        for idx, s in enumerate(series):
            ax = axes[idx // ncols][idx % ncols]
            for o in series:
                if o != s and len(fronts[o]) > 1:
                    ax.plot([r["x"] for r in fronts[o]], [r["y"] for r in fronts[o]],
                            color="#c8c8c8", lw=1.0, zorder=1)
            sub = [r for r in rows if r["series"] == s]
            draw_series(ax, sub, s, x_higher, y_higher, cfg.show_dominated, size=26)
            style_axes(ax, cfg, xlim, ylim, show_xlabel=(idx // ncols == nrows - 1 or idx + ncols >= len(series)),
                       show_ylabel=(idx % ncols == 0))
            ax.set_title(s, fontsize=10.5, color=PALETTE.get(s, "#333"), fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_size(9)
        for idx in range(len(series), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        # Prazni paneli: x oznaka ostaje na poslednjem vidljivom u koloni.
        for c in range(ncols):
            col_axes = [axes[r][c] for r in range(nrows) if r * ncols + c < len(series)]
            if col_axes:
                col_axes[-1].tick_params(labelbottom=True)
    else:
        sys.exit(f"Nepoznat mode '{cfg.mode}' (panels/multiples).")

    fig.tight_layout()
    out_path = save_fig(fig, get_results_path(f"{OUT_DIR}/{cfg.name}.png"))
    print(f"{cfg.name}: {len(rows)} modela iz {path.name} -> {out_path}")


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config, verbose=False)
