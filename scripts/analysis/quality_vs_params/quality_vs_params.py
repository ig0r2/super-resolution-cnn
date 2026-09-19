"""
quality_vs_params.py - grafik kvaliteta rekonstrukcije naspram velicine modela.

Crta dva panela jedan pored drugog za izabrane modele:
    levo   SSIM  naspram broja parametara (vise je bolje),
    desno  LPIPS naspram broja parametara (invertovana osa, nize je bolje).
Apscisa je logaritamska, tacke su obojene po arhitekturi.

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Sve konfiguracije se obradjuju u jednom pokretanju i daju odvojenu sliku
(imenovanu po "name" polju) u results/analysis/quality_vs_params/.
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

# Podrazumevano se izbacuju adversarijalni i JPEG modeli, jer na cistom skupu
# samo zaguse grafik. Postavi "exclude" na [] u konfiguraciji da ih ukljucis.
DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

# Fiksne boje/markeri za poznate arhitekture (radi konzistentnosti izmedju
# grafika); nepoznate dobijaju boju/marker iz rezervnog niza.
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
    "results": "results_2x_DIV2K_half.csv",  # CSV sa rezultatima (ime u results/ ili putanja)
    "out": None,                    # None -> ..._<name>.png
    "archs": None,                  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,                # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,                # None -> DEFAULT_EXCLUDE
    "min_params": None,
    "max_params": None,
    "label": None,                  # None -> izvedi iz imena fajla
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "all",
    },
    {
        "name": "standard",
        "archs": ["SRCNN", "VDSR", "EDSR", "IMDN", "RFDN"],
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
    """Iz imena fajla izvuce npr. 'DIV2K, x2' za oznaku na osama."""
    parts = Path(results_name).stem.split("_")
    factor = next((p[:-1] for p in parts if p.endswith("x") and p[:-1].isdigit()), None)
    dataset = next((p for p in parts if p in KNOWN_DATASETS), None)
    bits = [b for b in (dataset, f"$\\times${factor}" if factor else None) if b]
    return ", ".join(bits)


def load_rows(path: Path, cfg) -> list[dict]:
    """Ucita CSV i primeni sve filtere; vraca listu {arch, params, ssim, lpips}."""
    exclude = DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude
    archs = set(cfg.archs) if cfg.archs else None
    include = cfg.include or None

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
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
            ssim = to_float(r.get("SSIM"))
            lpips = to_float(r.get("LPIPS"))
            if not params:
                continue
            if cfg.min_params and params < cfg.min_params:
                continue
            if cfg.max_params and params > cfg.max_params:
                continue
            rows.append({"arch": arch, "params": params, "ssim": ssim, "lpips": lpips})
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


def plot_metric(rows, archs, key, label, dpi):
    """Jedan panel: metrika `key` (ssim/lpips) naspram broja parametara.

    Vraca figuru, ili None ako za tu metriku nema nijedne vrednosti.
    """
    colors, markers = style_for(archs)
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": dpi})
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    suffix = f" ({label})" if label else ""

    drawn = False
    for a in archs:
        xs = [r["params"] for r in rows if r["arch"] == a and r[key] is not None]
        ys = [r[key] for r in rows if r["arch"] == a and r[key] is not None]
        if not xs:
            continue
        ax.scatter(xs, ys, c=colors[a], marker=markers[a], s=40, alpha=0.85,
                   label=a, edgecolors="white", linewidths=0.4)
        drawn = True

    if not drawn:
        plt.close(fig)
        return None

    ax.set_xscale("log")
    ax.set_xlabel("Broj parametara (log)")

    if key == "ssim":
        ax.set_ylabel(f"SSIM{suffix}")
        ax.set_title("Kvalitet (SSIM) naspram veličine")
        ax.legend(loc="lower right", fontsize=9)
    else:
        ax.set_ylabel(f"LPIPS{suffix} — nize je bolje")
        ax.set_title("Perceptualni kvalitet (LPIPS) naspram veličine")
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


def main():
    if not CONFIGS:
        sys.exit("CONFIGS je prazna — dodaj bar jednu konfiguraciju.")
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

    rows = load_rows(results_path, cfg)
    if not rows:
        sys.exit("Nijedan model ne zadovoljava zadate filtere.")

    archs = ordered_archs(rows)
    label = cfg.label if cfg.label is not None else derive_label(cfg.results)

    # Osnova imena: cfg.out (bez .png) ako je zadat, inace izvedeno iz cfg.name.
    base = Path(cfg.out).stem if cfg.out else f"quality_vs_params_{cfg.name}"
    out_dir = get_results_path("analysis/quality_vs_params")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {a: sum(1 for r in rows if r["arch"] == a) for a in archs}
    print(f"Izvor: {results_path}")
    print(f"Modela: {len(rows)}  ({', '.join(f'{a}:{n}' for a, n in counts.items())})")

    # Dve odvojene slike: SSIM i LPIPS.
    for key, tag in (("ssim", "SSIM"), ("lpips", "LPIPS")):
        fig = plot_metric(rows, archs, key, label, cfg.dpi)
        if fig is None:
            print(f"Slika ({tag}): preskocena — nema vrednosti")
            continue
        out_path = out_dir / f"{base}_{tag}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Slika ({tag}): {out_path}")


if __name__ == "__main__":
    main()
