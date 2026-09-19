"""
fps_vs_params.py - grafik brzine inferencije naspram velicine modela.

Crta rasejani grafik na log-log osama: brzina (FPS) naspram broja parametara,
tacke obojene po arhitekturi. Log-log prikaz je odabran jer i broj parametara i
FPS obuhvataju vise redova velicine, a njihov odnos je priblizno stepeni (vise
parametara -> manje FPS), pa se na log-log osama vidi kao skoro prava linija.
Vodoravne isprekidane linije oznacavaju pragove realnog vremena (30 i 60 FPS).

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Sve konfiguracije se obradjuju u jednom pokretanju i daju odvojenu sliku
(imenovanu po "name" polju) u results/analysis/fps_vs_params_and_quality/.
"""

import csv
import math
import sys
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path import get_results_path

# Podrazumevano se izbacuju adversarijalni i JPEG modeli. Postavi "exclude" na []
# u konfiguraciji da ih ukljucis.
DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

# Fiksne boje/markeri za poznate arhitekture (konzistentno sa quality_vs_params).
PALETTE = {
    "SRCNN": "#7f7f7f", "VDSR": "#9467bd", "SRResNet": "#17becf",
    "EDSR": "#1b6ca8", "FastEDSR": "#e8702a", "IMDN": "#3c9a5f", "RFDN": "#b13b8f",
}
MARKERS = {
    "SRCNN": "P", "VDSR": "X", "SRResNet": "*",
    "EDSR": "o", "FastEDSR": "s", "IMDN": "^", "RFDN": "D",
}
PREFERRED_ORDER = ["SRCNN", "VDSR", "SRResNet", "EDSR", "FastEDSR", "IMDN", "RFDN"]

# Lepe vrednosti tikova na log osi FPS-a (crtaju se samo one u opsegu podataka).
FPS_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
PARAM_TICKS = [1e3, 1e4, 1e5, 1e6, 1e7]

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "default",  # koristi se za ime izlazne slike
    "results": "results_2x_FPS.csv",  # CSV sa merenjem brzine (ime u results/ ili putanja)
    "fps_col": "480p",  # kolona sa FPS vrednoscu (npr. 480p, 720p)
    "out": None,  # None -> ..._<name>.png
    "archs": None,  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,  # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,  # None -> DEFAULT_EXCLUDE
    "min_params": None,
    "max_params": None,
    "thresholds": [30, 60],  # pragovi FPS-a (vodoravne linije); [] za bez
    "label": None,  # None -> izvedi iz imena fajla
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "480p_all",
        "fps_col": "480p",
    },
    {
        "name": "480p_standard",
        "fps_col": "480p",
        "exclude": ["FastEDSR"]
    },
    {
        "name": "720p_all",
        "fps_col": "720p",
    },
    {
        "name": "720p_standard",
        "fps_col": "720p",
        "exclude": ["FastEDSR"]
    },
]


##############################################


def resolve_results(name: str) -> Path:
    """Prihvata golo ime (trazi u results/) ili punu/relativnu putanju."""
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return get_results_path(name)


def arch_of(model_name: str) -> str:
    """Arhitektura je prvi token iza opcionog `SR_` prefiksa."""
    parts = model_name.split("_")
    if parts and parts[0] == "SR":
        parts = parts[1:]
    return parts[0] if parts else model_name


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def derive_label(results_name: str) -> str:
    """Iz imena fajla izvuce npr. 'x2' za oznaku na osama."""
    parts = Path(results_name).stem.split("_")
    factor = next((p[:-1] for p in parts if p.endswith("x") and p[:-1].isdigit()), None)
    return f"$\\times${factor}" if factor else ""


def load_rows(path: Path, fps_col: str, cfg) -> list[dict]:
    """Ucita CSV i primeni filtere; vraca listu {arch, params, fps}."""
    exclude = DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude
    archs = set(cfg.archs) if cfg.archs else None
    include = cfg.include or None

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if fps_col not in reader.fieldnames:
            sys.exit(f"Kolona '{fps_col}' ne postoji. Dostupne: "
                     f"{', '.join(c for c in reader.fieldnames if c not in ('model_name', 'params', 'runtype'))}")
        for r in reader:
            name = r.get("model_name", "")
            if not name:
                continue
            if any(sub in name for sub in exclude):
                continue
            if include and not any(sub in name for sub in include):
                continue
            arch = arch_of(name)
            if archs and arch not in archs:
                continue
            params = to_float(r.get("params"))
            fps = to_float(r.get(fps_col))
            if not params or not fps:
                continue
            if cfg.min_params and params < cfg.min_params:
                continue
            if cfg.max_params and params > cfg.max_params:
                continue
            rows.append({"arch": arch, "params": params, "fps": fps})
    return rows


def style_for(archs: list[str]) -> tuple[dict, dict]:
    """Boja i marker po arhitekturi; nepoznate uzimaju iz rezervnog niza."""
    extra_c = cycle(["#d62728", "#17becf", "#bcbd22", "#8c564b", "#e377c2", "#2ca02c"])
    extra_m = cycle(["v", "<", ">", "p", "h", "8"])
    colors, markers = {}, {}
    for a in archs:
        colors[a] = PALETTE.get(a) or next(extra_c)
        markers[a] = MARKERS.get(a) or next(extra_m)
    return colors, markers


def ordered_archs(rows: list[dict]) -> list[str]:
    present = {r["arch"] for r in rows}
    known = [a for a in PREFERRED_ORDER if a in present]
    rest = sorted(present - set(known))
    return known + rest


def ticks_in_range(ticks, lo, hi):
    return [t for t in ticks if lo <= t <= hi]


def plot(rows, archs, fps_col, thresholds, label, dpi):
    colors, markers = style_for(archs)
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": dpi})
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    for a in archs:
        xs = [r["params"] for r in rows if r["arch"] == a]
        ys = [r["fps"] for r in rows if r["arch"] == a]
        if not xs:
            continue
        ax.scatter(xs, ys, c=colors[a], marker=markers[a], s=42, alpha=0.85,
                   label=a, edgecolors="white", linewidths=0.4)

    fps_vals = [r["fps"] for r in rows]
    par_vals = [r["params"] for r in rows]

    for thr in thresholds:
        ax.axhline(thr, color="#c0392b", ls="--", lw=0.9, zorder=1)
        ax.text(max(par_vals), thr, f" {thr:g} FPS", color="#c0392b",
                fontsize=8.5, va="bottom", ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    suffix = f" ({label})" if label else ""
    ax.set_xlabel("Broj parametara (log)")
    ax.set_ylabel(f"Brzina inferencije - {fps_col} (FPS, log){suffix}")
    ax.set_title("Brzina inferencije naspram velicine modela")

    xt = ticks_in_range(PARAM_TICKS, min(par_vals) * 0.9, max(par_vals) * 1.1)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"$10^{{{int(round(math.log10(t)))}}}$" for t in xt])
    yt = ticks_in_range(FPS_TICKS, min(fps_vals) * 0.9, max(fps_vals) * 1.1)
    ax.set_yticks(yt)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticklabels([f"{t:g}" for t in yt])

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


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
    results_path = resolve_results(cfg.results)
    if not results_path.exists():
        sys.exit(f"CSV ne postoji: {results_path}")

    rows = load_rows(results_path, cfg.fps_col, cfg)
    if not rows:
        sys.exit("Nijedan model ne zadovoljava zadate filtere.")

    archs = ordered_archs(rows)
    label = cfg.label if cfg.label is not None else derive_label(cfg.results)

    fig = plot(rows, archs, cfg.fps_col, cfg.thresholds, label, cfg.dpi)

    out_name = cfg.out or f"fps_vs_params_{cfg.name}.png"
    out_path = get_results_path("analysis/fps_vs_params_and_quality") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    counts = {a: sum(1 for r in rows if r["arch"] == a) for a in archs}
    print(f"Izvor: {results_path}  (kolona {cfg.fps_col})")
    print(f"Modela: {len(rows)}  ({', '.join(f'{a}:{n}' for a, n in counts.items())})")
    print(f"Slika:  {out_path}")


if __name__ == "__main__":
    main()
