"""
fp16_vs_fp32_speed_diff.py — koliko FP16 ubrzava inferencu u odnosu na FP32.

Uparuje modele po imenu izmedju FP32 i FP16 merenja brzine i za svaku rezoluciju
racuna ubrzanje (FPS_FP16 / FPS_FP32) po modelu. Ispisuje statistiku (min, prosek,
medijana, max ubrzanja, broj modela kod kojih je FP16 brzi) i crta rasejani grafik
ubrzanja naspram broja parametara (log osa) sa linijom na 1.0 koja razdvaja
ubrzanje (iznad) od usporenja (ispod). Vrednosti > 1 znace da je FP16 brzi.
"""

import csv
import math
import statistics
import sys
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.logger import Logger
from utils.path import get_results_path

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]
RES_COLORS = ["#1b6ca8", "#e8702a", "#3c9a5f", "#b13b8f"]
RES_MARKERS = ["o", "s", "^", "D"]
PARAM_TICKS = [1e3, 1e4, 1e5, 1e6, 1e7]

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


def resolve_results(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return get_results_path(name)


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


def zf(value, spec) -> str:
    """Formatira broj po `spec` sa decimalnim zarezom umesto tacke."""
    return format(value, spec).replace(".", ",")


def keep(name, exclude, archs, include):
    if not name:
        return False
    if any(sub in name for sub in exclude):
        return False
    if include and not any(sub in name for sub in include):
        return False
    if archs and arch_of(name) not in archs:
        return False
    return True


def load_side(path: Path, runtype, res_cols, exclude, archs, include, cfg):
    """Vrati {model_name: {"params": p, res: fps, ...}} za jednu precinost."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if runtype is not None and "runtype" not in reader.fieldnames:
            sys.exit(f"Kolona 'runtype' ne postoji u {path.name}, a zadat je runtype filter.")
        missing = [c for c in res_cols if c not in reader.fieldnames]
        if missing:
            usable = [c for c in reader.fieldnames if c not in ("model_name", "params", "runtype")]
            sys.exit(f"Nedostaju kolone {', '.join(missing)} u {path.name}. "
                     f"Dostupne: {', '.join(usable)}")
        for r in reader:
            name = r.get("model_name", "")
            if not keep(name, exclude, archs, include):
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
            for col in res_cols:
                v = to_float(r.get(col))
                if v:
                    rec[col] = v
            out[name] = rec
    return out


def main():
    if not CONFIGS:
        sys.exit("CONFIGS je prazna — dodaj bar jednu konfiguraciju.")
    for i, cfg_dict in enumerate(CONFIGS):
        cfg = SimpleNamespace(**{**CONFIG_DEFAULTS, **cfg_dict})
        if i:
            print("\n" + "=" * 92 + "\n")
        print(f"### Konfiguracija: {cfg.name}\n")
        run_config(cfg)


def run_config(cfg):
    exclude = DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude
    archs_filter = set(cfg.archs) if cfg.archs else None
    include = cfg.include or None

    fp16_path = resolve_results(cfg.fp16_results)
    fp32_path = resolve_results(cfg.fp32_results)
    for p in (fp16_path, fp32_path):
        if not p.exists():
            sys.exit(f"CSV ne postoji: {p}")

    # Ako obe precinosti dolaze iz istog fajla, moraju se razlikovati po runtype.
    if fp16_path == fp32_path and cfg.fp16_runtype == cfg.fp32_runtype:
        sys.exit("FP16 i FP32 dolaze iz istog CSV-a bez razlicitog runtype filtera. "
                 "Zadaj fp16_runtype i fp32_runtype, ili odvojene "
                 "fp16_results / fp32_results.")

    fp16 = load_side(fp16_path, cfg.fp16_runtype, cfg.res, exclude, archs_filter, include, cfg)
    fp32 = load_side(fp32_path, cfg.fp32_runtype, cfg.res, exclude, archs_filter, include, cfg)
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
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": cfg.dpi})
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
    xt = [t for t in PARAM_TICKS if min(all_p) * 0.9 <= t <= max(all_p) * 1.1]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"$10^{{{int(round(math.log10(t)))}}}$" for t in xt])
    ax.set_xlabel("Broj parametara (log)")
    ax.set_ylabel("ubrzanje (FPS FP16 / FPS FP32)")
    ax.set_title("Ubrzanje FP16 nad FP32")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path = get_results_path("analysis/fp16_vs_fp32") / plot_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
