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
(imenovanu po "name" polju) u results/analysis/fps_vs_params_and_quality/.
"""

import csv
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

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

PALETTE = {
    "SRCNN": "#7f7f7f", "VDSR": "#9467bd", "SRResNet": "#17becf",
    "EDSR": "#1b6ca8", "FastEDSR": "#e8702a", "IMDN": "#3c9a5f", "RFDN": "#b13b8f",
}
MARKERS = {
    "SRCNN": "P", "VDSR": "X", "SRResNet": "*",
    "EDSR": "o", "FastEDSR": "s", "IMDN": "^", "RFDN": "D",
}
PREFERRED_ORDER = ["SRCNN", "VDSR", "SRResNet", "EDSR", "FastEDSR", "IMDN", "RFDN"]

FPS_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
KNOWN_DATASETS = ("Set5", "Set14", "BSD100", "Urban100", "DIV2K")

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "default",              # koristi se za ime izlazne slike
    "fps_results": "results_2x_FPS.csv",         # CSV sa merenjem brzine
    "quality_results": "results_2x_DIV2K_half.csv",  # CSV sa merenjem kvaliteta
    "fps_col": "720p",              # kolona sa FPS vrednoscu (npr. 720p, 480p)
    "metric": "SSIM",               # SSIM/PSNR: vise je bolje, LPIPS: nize je bolje
    "out": None,                    # None -> ..._<name>.png
    "archs": None,                  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,                # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,                # None -> DEFAULT_EXCLUDE
    "min_params": None,
    "max_params": None,
    "thresholds": [30, 60],         # pragovi FPS-a (vertikalne linije); [] za bez
    "label": None,                  # None -> izvedi iz imena fajlova
    "no_pareto": False,             # True -> bez linije Pareto fronta
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "SSIM_720p",
        "metric": "SSIM",
        "fps_col": "720p",
    },
    {
        "name": "SSIM_480p",
        "metric": "SSIM",
        "fps_col": "480p",
    },
    {
        "name": "LPIPS_720p",
        "metric": "LPIPS",
        "fps_col": "720p",
    },
    {
        "name": "LPIPS_480p",
        "metric": "LPIPS",
        "fps_col": "480p",
    },
]


##############################################


def resolve_results(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return get_results_path(name)


def arch_of(model_name: str) -> str:
    parts = model_name.split("_")
    if parts and parts[0] == "SR":
        parts = parts[1:]
    return parts[0] if parts else model_name


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def derive_label(*names) -> str:
    parts = []
    for n in names:
        parts += Path(n).stem.split("_")
    factor = next((p[:-1] for p in parts if p.endswith("x") and p[:-1].isdigit()), None)
    dataset = next((p for p in parts if p in KNOWN_DATASETS), None)
    bits = [b for b in (dataset, f"$\\times${factor}" if factor else None) if b]
    return ", ".join(bits)


def read_quality(path: Path, metric: str) -> dict:
    """Mapa model_name -> vrednost metrike."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if metric not in reader.fieldnames:
            sys.exit(f"Metrika '{metric}' ne postoji u {path.name}.")
        for r in reader:
            v = to_float(r.get(metric))
            if r.get("model_name") and v is not None:
                out[r["model_name"]] = v
    return out


def load_rows(fps_path, quality_path, cfg) -> list[dict]:
    """Spoji brzinu i kvalitet po imenu modela i primeni filtere."""
    exclude = DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude
    archs = set(cfg.archs) if cfg.archs else None
    include = cfg.include or None
    quality = read_quality(quality_path, cfg.metric)

    rows = []
    with open(fps_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if cfg.fps_col not in reader.fieldnames:
            sys.exit(f"Kolona '{cfg.fps_col}' ne postoji u {fps_path.name}. Dostupne: "
                     f"{', '.join(c for c in reader.fieldnames if c not in ('model_name', 'params', 'runtype'))}")
        for r in reader:
            name = r.get("model_name", "")
            if not name or name not in quality:
                continue
            if any(sub in name for sub in exclude):
                continue
            if include and not any(sub in name for sub in include):
                continue
            arch = arch_of(name)
            if archs and arch not in archs:
                continue
            params = to_float(r.get("params"))
            fps = to_float(r.get(cfg.fps_col))
            if not fps:
                continue
            if params and cfg.min_params and params < cfg.min_params:
                continue
            if params and cfg.max_params and params > cfg.max_params:
                continue
            rows.append({"name": name, "arch": arch, "fps": fps, "q": quality[name]})
    return rows


def pareto_front(rows: list[dict], maximize: bool) -> list[dict]:
    """Nedominirane tacke: veci FPS i bolji kvalitet (bolji = veci ako maximize)."""
    def better_eq(a, b):
        return a >= b if maximize else a <= b

    def strictly(a, b):
        return a > b if maximize else a < b

    front = []
    for p in rows:
        dominated = any(
            o is not p and o["fps"] >= p["fps"] and better_eq(o["q"], p["q"])
            and (o["fps"] > p["fps"] or strictly(o["q"], p["q"]))
            for o in rows
        )
        if not dominated:
            front.append(p)
    front.sort(key=lambda r: r["fps"])
    return front


def style_for(archs):
    extra_c = cycle(["#d62728", "#17becf", "#bcbd22", "#8c564b", "#e377c2", "#2ca02c"])
    extra_m = cycle(["v", "<", ">", "p", "h", "8"])
    colors, markers = {}, {}
    for a in archs:
        colors[a] = PALETTE.get(a) or next(extra_c)
        markers[a] = MARKERS.get(a) or next(extra_m)
    return colors, markers


def ordered_archs(rows):
    present = {r["arch"] for r in rows}
    known = [a for a in PREFERRED_ORDER if a in present]
    return known + sorted(present - set(known))


def plot(rows, archs, cfg, label, maximize):
    colors, markers = style_for(archs)
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": cfg.dpi})
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    for a in archs:
        xs = [r["fps"] for r in rows if r["arch"] == a]
        ys = [r["q"] for r in rows if r["arch"] == a]
        if not xs:
            continue
        ax.scatter(xs, ys, c=colors[a], marker=markers[a], s=42, alpha=0.85,
                   label=a, edgecolors="white", linewidths=0.4)

    if not cfg.no_pareto:
        front = pareto_front(rows, maximize)
        ax.plot([r["fps"] for r in front], [r["q"] for r in front],
                "-", color="#333", lw=1.3, zorder=1, label="Pareto front")

    for thr in cfg.thresholds:
        ax.axvline(thr, color="#aaa", ls="--", lw=0.9, zorder=0)
        ax.text(thr, min(r["q"] for r in rows), f" {thr:g} FPS", rotation=90,
                color="#888", fontsize=8.5, va="bottom", ha="left")

    ax.set_xscale("log")
    suffix = f" ({label})" if label else ""
    better = "vise je bolje" if maximize else "nize je bolje"
    ax.set_xlabel(f"Brzina inferencije - {cfg.fps_col} (FPS, log)")
    ax.set_ylabel(f"{cfg.metric}{suffix} - {better}")
    ax.set_title(f"Odnos brzine i kvaliteta ({cfg.metric})")
    if not maximize:
        ax.invert_yaxis()

    lo, hi = min(r["fps"] for r in rows), max(r["fps"] for r in rows)
    xt = [t for t in FPS_TICKS if lo * 0.9 <= t <= hi * 1.1]
    ax.set_xticks(xt)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticklabels([f"{t:g}" for t in xt])

    loc = "lower left" if maximize else "upper right"
    ax.legend(loc=loc, fontsize=9, framealpha=0.9)
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
    fps_path = resolve_results(cfg.fps_results)
    quality_path = resolve_results(cfg.quality_results)
    for p in (fps_path, quality_path):
        if not p.exists():
            sys.exit(f"CSV ne postoji: {p}")

    rows = load_rows(fps_path, quality_path, cfg)
    if not rows:
        sys.exit("Nijedan model nema i brzinu i kvalitet uz zadate filtere.")

    archs = ordered_archs(rows)
    label = cfg.label if cfg.label is not None else derive_label(cfg.quality_results, cfg.fps_results)
    maximize = cfg.metric != "LPIPS"

    fig = plot(rows, archs, cfg, label, maximize)

    out_name = cfg.out or f"fps_vs_quality_{cfg.name}.png"
    out_path = get_results_path("analysis/fps_vs_params_and_quality") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    counts = {a: sum(1 for r in rows if r["arch"] == a) for a in archs}
    front = pareto_front(rows, maximize)
    print(f"Brzina:  {fps_path}  (kolona {cfg.fps_col})")
    print(f"Kvalitet:{quality_path}  (metrika {cfg.metric})")
    print(f"Modela:  {len(rows)}  ({', '.join(f'{a}:{n}' for a, n in counts.items())})")
    print(f"Pareto front ({len(front)}): {', '.join(r['name'] for r in front)}")
    print(f"Slika:   {out_path}")


if __name__ == "__main__":
    main()
