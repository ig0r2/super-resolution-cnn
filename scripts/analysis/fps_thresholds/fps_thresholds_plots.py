"""
fps_thresholds_plots.py - Pareto front kvalitet/brzina po arhitekturi (uz pragove FPS-a).

Grafik uz tabele iz fps_thresholds_tables.py: x osa je brzina (FPS, log), y osa
metrika, a za svaku seriju se crta Pareto front (veci FPS i bolji kvalitet).
Vertikalne linije oznacavaju pragove FPS-a. Dva rezima prikaza:

  - "arch": front po seriji, opciono podeljen u panele (npr. HR prostor,
    destilacija, rezidual nad uvecanim ulazom) da grafik ne bi bio "spaghetti";
  - "global": jedan zajednicki front preko svih serija; tacke su obojene po
    seriji kojoj pripadaju.

Serija je ime arhitekture iz model_name (SR_EDSR_2x_16_64 -> "EDSR"). Viseskalni
modeli (bez Nx tokena u imenu) dobijaju sufiks "-M" ("FastEDSR-M") i ulaze samo
uz include_multiscale=True.

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Slike se cuvaju u results/analysis/fps_thresholds/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.plotting import (EXTRA_COLORS, FPS_TICKS, MARKERS, PALETTE, pareto_front,
                                     save_fig, set_fps_ticks, setup_style)
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from utils.analysis.config import run_configs
from utils.analysis.data import read_rows, require_results, to_float
from utils.analysis.fmt import zf
from utils.analysis.metrics import check_metrics, higher_is_better
from utils.analysis.names import INTERPOLATIONS, series_name
from utils.path import get_results_path

OUT_DIR = "analysis/fps_thresholds"

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

CONFIG_DEFAULTS = {
    "name": "default",  # ime izlazne slike (fps_thresholds_<name>.png)
    "results": "results_2x_DIV2K_half.csv",
    "fps_col": "FPS 480p",  # kolona sa brzinom (x osa)
    "metric": "LPIPS",  # metrika (y osa)
    "mode": "arch",  # "arch" (front po seriji) ili "global" (jedan front)
    # Lista panela {"title", "series"}; jedan panel sa "series": None prikazuje sve serije.
    "panels": [{"title": None, "series": None}],
    "line": "linear",  # "linear" ili "step" (najbolji kvalitet uz FPS >= x)
    "exclude": None,  # None -> DEFAULT_EXCLUDE
    "include_multiscale": False,  # True -> ukljuci i viseskalne modele (serije "-M")
    "show_dominated": False,  # blede tacke van fronta
    "reference": "bicubic",  # horizontalna linija za interpolaciju; None za bez
    "thresholds": [],  # vertikalne linije
    "label": "DIV2K, ×2",  # dodaje se oznaci ose metrike
    "figsize": None,  # None -> automatski po broju panela
    "font_size": 10.5,
    "marker_size": 40,  # velicina tacaka na frontu (mode="arch")
    "fps_ticks": FPS_TICKS,  # oznake na FPS osi (proredi za uske panele)
    "dpi": 140,
}

##############################################

LITERATURE = ["VDSR", "EDSR", "IMDN", "RFDN"]
# Podela po prostoru i nacinu izracunavanja; EDSR je referenca u svakoj grupi
# (LR prostor, obicne konvolucije, bez reziduala nad uvecanim ulazom).
HR_GROUP = {"title": "HR prostor", "series": ["EDSR", "VDSR"]}
DISTILL_GROUP = {"title": "LR: destilacija i pažnja", "series": ["EDSR", "IMDN", "RFDN"]}
GROUP_STYLE = {"figsize": (9.0, 3.4), "font_size": 8.5, "marker_size": 20, "reference": None,
               "fps_ticks": [1, 10, 100, 1000]}  # tri panela u sirini teksta
PLAIN_SINGLE = ["EDSR", "FastEDSR", "ABPN"]

CONFIGS = []
for _metric in ("LPIPS", "SSIM"):
    CONFIGS += [
        {"name": f"{_metric}_panels_single", "metric": _metric, **GROUP_STYLE,
         "panels": [HR_GROUP, DISTILL_GROUP,
                    {"title": "LR: rezidual nad uvećanim ulazom", "series": PLAIN_SINGLE}]},
    ]
CONFIGS += [
    {"name": "LPIPS_global_single", "metric": "LPIPS", "mode": "global", "show_dominated": True,
     "panels": [{"title": None, "series": LITERATURE + PLAIN_SINGLE[1:]}]},
]


##############################################


def base_arch(series):
    return series.split("-")[0].split("_")[0]


def load_rows(cfg):
    """Lista {name, series, config, fps, y} + mapa interpolacija -> vrednost metrike."""
    path = require_results(cfg.results)
    exclude = DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude
    rows, interp = [], {}
    for r in read_rows(path, [cfg.fps_col, cfg.metric]):
        name = r.get("model_name", "")
        y = to_float(r.get(cfg.metric))
        if y is None:
            continue
        if name in INTERPOLATIONS:
            interp[name] = y
            continue
        if not name.startswith("SR_") or any(s in name for s in exclude):
            continue
        series, config = series_name(name)
        if series.endswith("-M") and not cfg.include_multiscale:
            continue
        fps = to_float(r.get(cfg.fps_col))
        if not fps:
            continue
        rows.append({"name": name, "series": series, "config": config, "fps": fps, "y": y})
    return path, rows, interp


def panel_rows(rows, series, include_multiscale):
    """Redovi panela i serije koje u njemu stvarno postoje (tim redom)."""
    present = {r["series"] for r in rows}
    if series is None:
        return rows, sorted(present)
    for s in series:
        if s not in present and not (s.endswith("-M") and not include_multiscale):
            print(f"  [upozorenje] serija '{s}' nema nijedan model sa FPS vrednoscu - preskacem")
    return [r for r in rows if r["series"] in series], [s for s in series if s in present]


def front_of(rows, higher):
    """Nedominirane tacke: veci FPS i bolji kvalitet."""
    return pareto_front(rows, x_higher=True, y_higher=higher, x="fps", y="y")


def styles(series_list):
    extra = iter(EXTRA_COLORS * 3)
    colors, markers, dashed = {}, {}, {}
    for s in series_list:
        a = base_arch(s)
        colors[s] = PALETTE.get(a) or next(extra)
        markers[s] = MARKERS.get(a, "o")
        # Varijanta iste arhitekture u istom panelu (npr. FastEDSR i FastEDSR-M) - isprekidano.
        dashed[s] = s != a and a in series_list
    return colors, markers, dashed


def draw_line(ax, front, color, line, dashed, xmin):
    fx = [r["fps"] for r in front]
    fy = [r["y"] for r in front]
    ls = "--" if dashed else "-"
    if line == "step":
        # Najbolji kvalitet uz FPS >= x: vrednost tacke vazi i za sve x <= njenog FPS-a.
        ax.step([xmin] + fx, [fy[0]] + fy, where="pre", color=color, lw=1.6, ls=ls, alpha=0.9)
    else:
        ax.plot(fx, fy, color=color, lw=1.6, ls=ls, alpha=0.9)


def draw_panel(ax, cfg, rows, series, colors, markers, dashed, xlim, higher):
    if cfg.mode == "global":
        for s in series:
            sub = [r for r in rows if r["series"] == s]
            ax.scatter([r["fps"] for r in sub], [r["y"] for r in sub], c=colors[s],
                       marker=markers[s], s=26, alpha=0.25 if cfg.show_dominated else 0, edgecolors="none")
        front = front_of(rows, higher)
        draw_line(ax, front, "#333", cfg.line, False, xlim[0])
        for s in series:
            pts = [r for r in front if r["series"] == s]
            # Varijanta iste arhitekture (npr. FastEDSR-M uz FastEDSR) - prazan marker.
            face = "white" if dashed[s] else colors[s]
            ax.scatter([r["fps"] for r in pts], [r["y"] for r in pts], facecolors=face,
                       edgecolors=colors[s], marker=markers[s], s=48, linewidths=1.2, zorder=3)
            ax.scatter([], [], facecolors=face, edgecolors=colors[s], marker=markers[s], s=40,
                       linewidths=1.2, label=s if pts else f"{s} (van fronta)")
        # Globalni front ide preko cele sirine - legenda van grafika.
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, framealpha=0.9)
    else:
        for s in series:
            sub = [r for r in rows if r["series"] == s]
            front = front_of(sub, higher)
            if cfg.show_dominated:
                rest = [r for r in sub if r not in front]
                ax.scatter([r["fps"] for r in rest], [r["y"] for r in rest], c=colors[s],
                           marker=markers[s], s=cfg.marker_size * 0.55, alpha=0.2, edgecolors="none")
            draw_line(ax, front, colors[s], cfg.line, dashed[s], xlim[0])
            ax.scatter([r["fps"] for r in front], [r["y"] for r in front], c=colors[s],
                       marker=markers[s], s=cfg.marker_size, edgecolors="white", linewidths=0.5,
                       zorder=3, label=s)
        ax.legend(loc="lower left" if higher else "upper right", fontsize=cfg.font_size - 2, framealpha=0.9)


def style_axes(ax, cfg, rows, ref, xlim):
    # Referentna interpolacija se crta samo ako je blizu opsega podataka, da ne bi
    # rastegla osu (npr. bikubni LPIPS je daleko ispod svih modela).
    ys = [r["y"] for r in rows]
    margin = 0.1 * (max(ys) - min(ys))
    if ref is not None and min(ys) - margin <= ref <= max(ys) + margin:
        ax.axhline(ref, color="#999", ls=":", lw=1.0, zorder=0)
        ax.text(xlim[0], ref, f" {cfg.reference}", color="#777", fontsize=8, va="bottom", ha="left")
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for thr in cfg.thresholds:
        ax.axvline(thr, color="#aaa", ls="--", lw=0.9, zorder=0)
        ax.text(thr, 0.02, f" {zf(thr, 'g')} FPS", transform=trans, rotation=90, color="#888",
                fontsize=8, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    set_fps_ticks(ax, *xlim, ticks=cfg.fps_ticks)
    ax.set_xlabel(f"Brzina inferencije - {cfg.fps_col} (log)")


def run_config(cfg):
    check_metrics([cfg.metric])
    if cfg.mode not in ("arch", "global"):
        sys.exit(f"Nepoznat mode '{cfg.mode}' (arch/global).")
    higher = higher_is_better(cfg.metric)
    path, all_rows, interp = load_rows(cfg)

    panels = []
    for p in cfg.panels:
        rows, series = panel_rows(all_rows, p.get("series"), cfg.include_multiscale)
        if rows:
            panels.append((p, rows, series))
    if not panels:
        sys.exit("Nijedan panel nema modela.")

    used = [r for _, rows, _ in panels for r in rows]
    xlim = (min(r["fps"] for r in used) * 0.8, max(r["fps"] for r in used) * 1.25)
    every = []
    for _, _, series in panels:
        every += [s for s in series if s not in every]
    colors, markers, dashed = styles(every)
    ref = interp.get(cfg.reference) if cfg.reference else None

    setup_style(cfg.dpi, cfg.font_size)
    n = len(panels)
    figsize = cfg.figsize or ((9.0 if cfg.mode == "global" else 7.6, 5.2) if n == 1 else (5.6 * n, 4.8))
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True, squeeze=False)
    for ax, (p, rows, series) in zip(axes[0], panels):
        draw_panel(ax, cfg, rows, series, colors, markers, dashed, xlim, higher)
        style_axes(ax, cfg, rows, ref, xlim)
        if p.get("title"):
            ax.set_title(p["title"], fontsize=cfg.font_size + 0.5)
    axes[0][0].set_ylabel(f"{cfg.metric} ({cfg.label}) - {'više' if higher else 'niže'} je bolje")
    if not higher:
        axes[0][0].invert_yaxis()  # sharey -> vazi za sve panele

    fig.tight_layout()
    out_path = save_fig(fig, get_results_path(f"{OUT_DIR}/fps_thresholds_{cfg.name}.png"))

    for p, rows, series in panels:
        title = p.get("title") or "sve"
        if cfg.mode == "global":
            front = front_of(rows, higher)
            print(f"[{title}] globalni front: {', '.join(f'{r['series']} {r['config']}' for r in front)}")
        else:
            for s in series:
                front = front_of([r for r in rows if r["series"] == s], higher)
                print(f"[{title}] {s}: {', '.join(f'{r['config']}@{r['fps']:.0f}' for r in front)}")
    print(f"{cfg.name}: {len({r['name'] for r in used})} modela iz {path.name} -> {out_path}")


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config, verbose=False)
