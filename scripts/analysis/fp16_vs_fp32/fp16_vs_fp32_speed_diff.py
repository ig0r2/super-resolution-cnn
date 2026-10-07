"""
fp16_vs_fp32_speed_diff.py — koliko FP16 ubrzava inferencu u odnosu na FP32.

Uparuje modele po imenu izmedju FP32 i FP16 merenja brzine i za svaku rezoluciju
racuna ubrzanje (FPS_FP16 / FPS_FP32) po modelu. Ispisuje statistiku (min, prosek,
medijana, max ubrzanja, broj modela kod kojih je FP16 brzi) i crta rasejani grafik
ubrzanja naspram broja parametara (log osa) sa linijom na 1.0 koja razdvaja
ubrzanje (iznad) od usporenja (ispod). Vrednosti > 1 znace da je FP16 brzi.
"""

import statistics
import sys
from itertools import cycle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.plotting import save_fig, set_param_ticks, setup_style
import matplotlib.pyplot as plt

from utils.analysis.config import run_configs
from utils.analysis.data import name_filter, read_rows, require_results, to_float
from utils.analysis.fmt import zf
from utils.logger import Logger
from utils.path import get_results_path

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]
RES_COLORS = ["#1b6ca8", "#e8702a", "#3c9a5f", "#b13b8f"]
RES_MARKERS = ["o", "s", "^", "D"]

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "default",  # koristi se za imena izlaznih fajlova
    "fp16_results": "results_2x_FPS.csv",  # CSV sa FP16 merenjima brzine
    "fp32_results": "results_2x_FPS_fp32.csv",  # CSV sa FP32 merenjima brzine
    "fp16_runtype": None,  # vrednost kolone runtype za FP16 (uz isti CSV)
    "fp32_runtype": None,  # vrednost kolone runtype za FP32 (uz isti CSV)
    "res": ["480p"],  # rezolucije (kolone) za poredjenje
    "archs": None,  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,  # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,  # None -> DEFAULT_EXCLUDE
    "min_params": None,
    "max_params": None,
    "out": None,  # None -> results/analysis/fp16_vs_fp32/..._<name>.txt
    "plot_out": None,  # None -> ..._<name>.png
    "no_log": False,  # True -> samo ispis na ekran, bez fajla
    "no_plot": False,  # True -> bez grafika
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "480p",
        "res": ["480p"],
    },
    {
        "name": "480p_720p",
        "res": ["480p", "720p"],
    },
]


##############################################


def load_side(path: Path, runtype, keep, cfg):
    """Vrati {model_name: {"params": p, res: fps, ...}} za jednu precinost."""
    out = {}
    required = list(cfg.res) + (["runtype"] if runtype is not None else [])
    for r in read_rows(path, required):
        name = r.get("model_name", "")
        if not keep(name):
            continue
        if runtype is not None and r.get("runtype", "") != runtype:
            continue
        params = to_float(r.get("params"))
        if not params:
            continue
        if cfg.min_params and params < cfg.min_params:
            continue
        if cfg.max_params and params > cfg.max_params:
            continue
        rec = {"params": params}
        for col in cfg.res:
            v = to_float(r.get(col))
            if v:
                rec[col] = v
        out[name] = rec
    return out


def run_config(cfg):
    keep = name_filter(DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude, cfg.archs, cfg.include)
    fp16_path = require_results(cfg.fp16_results)
    fp32_path = require_results(cfg.fp32_results)

    # Ako obe precinosti dolaze iz istog fajla, moraju se razlikovati po runtype.
    if fp16_path == fp32_path and cfg.fp16_runtype == cfg.fp32_runtype:
        sys.exit("FP16 i FP32 dolaze iz istog CSV-a bez razlicitog runtype filtera. "
                 "Zadaj fp16_runtype i fp32_runtype, ili odvojene "
                 "fp16_results / fp32_results.")

    fp16 = load_side(fp16_path, cfg.fp16_runtype, keep, cfg)
    fp32 = load_side(fp32_path, cfg.fp32_runtype, keep, cfg)
    common = sorted(set(fp16) & set(fp32))
    if not common:
        sys.exit("Nema modela prisutnih u obe precinosti (proveri runtype/fajlove/filtre).")

    # {res: [(params, ubrzanje, ime), ...]}
    series = {res: [] for res in cfg.res}
    for name in common:
        for res in cfg.res:
            a = fp16[name].get(res)
            b = fp32[name].get(res)
            if a and b:
                series[res].append((fp16[name]["params"], a / b, name))
    if not any(series.values()):
        sys.exit("Nema uparenih FPS vrednosti ni za jednu rezoluciju.")

    # Sazetak po rezoluciji: min/prosek/medijana/max ubrzanja.
    summary = []  # (res, n, min, mean, median, max, faster_n, best_model, worst_model)
    for res in cfg.res:
        pts = series[res]
        if not pts:
            continue
        vals = [r for _, r, _ in pts]
        best = max(pts, key=lambda t: t[1])
        worst = min(pts, key=lambda t: t[1])
        faster_n = sum(1 for v in vals if v > 1.0)
        summary.append((res, len(vals), min(vals), sum(vals) / len(vals),
                        statistics.median(vals), max(vals), faster_n,
                        best[2], worst[2]))

    # Tekstualni izlaz se opciono cuva u results/analysis/ preko Logger-a.
    if cfg.no_log:
        report(summary, fp16_path, fp32_path, len(common))
    else:
        out_path = Path(cfg.out) if cfg.out else get_results_path(
            f"analysis/fp16_vs_fp32/fp16_vs_fp32_speed_diff_{cfg.name}.txt")
        with Logger(out_path):
            report(summary, fp16_path, fp32_path, len(common))

    if not cfg.no_plot:
        plot_name = cfg.plot_out or f"fp16_vs_fp32_speed_diff_{cfg.name}.png"
        plot_path = plot(series, cfg, plot_name)
        print(f"Slika: {plot_path}")


def report(summary, fp16_path, fp32_path, n_common):
    print(f"FP16 izvor: {fp16_path.name}")
    print(f"FP32 izvor: {fp32_path.name}")
    print(f"Uparenih modela: {n_common}  (ubrzanje = FPS FP16 / FPS FP32)")
    print()
    header = (f"{'Rez.':<8} {'N':>4} {'min':>8} {'prosek':>8} {'medijana':>10} "
              f"{'max':>8} {'FP16 brzi':>10}   najvece / najmanje ubrzanje")
    print(header)
    print("-" * len(header))
    for res, n, mn, mean, med, mx, faster_n, best_model, worst_model in summary:
        print(f"{res:<8} {n:>4} {zf(mn, '>8.2f')} {zf(mean, '>8.2f')} "
              f"{zf(med, '>10.2f')} {zf(mx, '>8.2f')} "
              f"{faster_n:>10}   {best_model} / {worst_model}")


def plot(series, cfg, plot_name):
    """Rasejani grafik ubrzanja (FP16/FP32) naspram broja parametara (log osa)."""
    setup_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    colors = cycle(RES_COLORS)
    markers = cycle(RES_MARKERS)

    all_p = []
    for res in cfg.res:
        pts = sorted((p, r) for p, r, _ in series[res])
        if not pts:
            continue
        xs = [p for p, _ in pts]
        ys = [r for _, r in pts]
        all_p += xs
        ax.scatter(xs, ys, s=40, alpha=0.85, color=next(colors), marker=next(markers),
                   label=res, edgecolors="white", linewidths=0.4)

    ax.axhline(1.0, color="#c0392b", ls="--", lw=1.1)
    ax.text(0.99, 1.02, "FP16 brzi (iznad linije)", transform=ax.get_yaxis_transform(),
            color="#c0392b", fontsize=8.5, ha="right", va="bottom")

    ax.set_xscale("log")
    set_param_ticks(ax, all_p)
    ax.set_xlabel("Broj parametara (log)")
    ax.set_ylabel("ubrzanje (FPS FP16 / FPS FP32)")
    ax.set_title("Ubrzanje FP16 nad FP32")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    return save_fig(fig, get_results_path("analysis/fp16_vs_fp32") / plot_name)


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
