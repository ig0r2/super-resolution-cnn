"""
runtimes_compare.py - poredjenje runtime backend-a na video merenju (ms).

Cita results_2x_video_ms.csv i za izabranu metriku (npr. SR) crta grupisani
horizontalni bar grafik. Bez "runtypes" filtera uzimaju se svi runtype-ovi koji
pocinju sa "onnxruntime-"; sa filterom se uzimaju tacno zadati (npr. i native
"tensorrt", "ncnn-vulkan"). Za svaki runtype se uzima prosek vrednosti preko
modela koji su zajednicki za sve ukljucene backend-e (fer poredjenje na istom
skupu modela). Nazivi runtype-ova se mapiraju na citljive oznake (ORT-TensorRT,
ORT-CUDA, ORT-DirectML, TensorRT, ncnn-Vulkan).

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Sve konfiguracije se obradjuju u jednom pokretanju i daju odvojenu sliku
(imenovanu po "name" polju) u results/analysis/runtimes/.
"""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path import get_results_path

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

# Redosled i citljive oznake za runtype-ove (koristi se i kao podrazumevani
# redosled kad "runtypes" nije zadat).
RUNTYPE_LABELS = {
    "onnxruntime-cuda": "ORT-CUDA",
    "onnxruntime-directml": "ORT-DirectML",
    "onnxruntime-tensorrt": "ORT-TensorRT",
    "tensorrt": "TensorRT",
    "ncnn-vulkan": "ncnn-Vulkan",
}
RUNTYPE_COLORS = {
    "onnxruntime-tensorrt": "#1b6ca8",
    "onnxruntime-cuda": "#3c9a5f",
    "onnxruntime-directml": "#e8702a",
    "tensorrt": "#b13b8f",
    "ncnn-vulkan": "#9467bd",
}

# Kratke oznake metrika -> deo imena kolone (spaja se sa "res": f"{res} {key}").
METRIC_LABELS = {
    "decode": "Dekodovanje",
    "sr": "SR",
    "display": "Prikaz",
    "total": "Ukupno",
}

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "sr",  # koristi se za ime izlazne slike
    "results": "results_2x_video_ms.csv",  # CSV sa video merenjem (ms)
    "res": "480p",  # prefiks rezolucije u imenu kolone
    "unit": "ms",  # "ms" (nize je bolje) ili "fps" (1000/ms, vise je bolje)
    "runtypes": None,  # None -> svi onnxruntime-*; ili lista tacnih runtype-ova
    # (tim redom), npr. ["onnxruntime-tensorrt", "tensorrt"]
    "metrics": ["sr"],  # metrike (grupe na y-osi): decode/sr/display/total
    "group_by": "metric",  # "metric" -> grupe = metrike; "size" -> grupe = velicine
    "size_bins": [         # koristi se samo kad je group_by == "size" (prva metrika)
        ("Mali\nmodeli", 0, 100_000),          # params u [0, 100k)
        ("Veliki\nmodeli", 100_000, None),     # params >= 100k (None = bez gornje granice)
    ],
    "archs": None,  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,  # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,  # None -> DEFAULT_EXCLUDE
    "out": None,  # None -> runtimes_<name>.png
    "legend_loc": "upper left",  # pozicija legende (matplotlib loc), "upper left", "lower right", "best"
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "sr",
        "metrics": ["sr"],
        "unit": "ms",
    },
    {
        "name": "sr",
        "metrics": ["sr"],
        "unit": "fps",
    },
    {
        "name": "sr_by_size",
        "metrics": ["sr"],
        "unit": "fps",
        "group_by": "size",
    },
    {
        "name": "ort_vs_trt",
        "metrics": ["sr"],
        "unit": "ms",
        "runtypes": ["onnxruntime-tensorrt", "tensorrt"],
    },
    {
        "name": "ort_vs_trt_total",
        "metrics": ["total"],
        "unit": "ms",
        "runtypes": ["onnxruntime-tensorrt", "tensorrt"],
    },
    {
        "name": "ort_vs_trt",
        "metrics": ["sr"],
        "unit": "fps",
        "runtypes": ["onnxruntime-tensorrt", "tensorrt"],
    },
    {
        "name": "ort_vs_trt_by_size",
        "metrics": ["sr"],
        "unit": "fps",
        "group_by": "size",
        "runtypes": ["onnxruntime-tensorrt", "tensorrt"],
    },
    {
        "name": "ort_vs_trt_total",
        "metrics": ["total"],
        "unit": "fps",
        "runtypes": ["onnxruntime-tensorrt", "tensorrt"],
    },
    {
        "name": "ort_vs_trt_vs_ncnn",
        "metrics": ["sr"],
        "unit": "ms",
        "runtypes": ["onnxruntime-tensorrt", "tensorrt", "ncnn-vulkan"],
        "legend_loc": "upper right"
    },
    {
        "name": "ort_vs_trt_vs_ncnn",
        "metrics": ["sr"],
        "unit": "fps",
        "runtypes": ["onnxruntime-tensorrt", "tensorrt", "ncnn-vulkan"],
    },
    {
        "name": "all",
        "metrics": ["sr"],
        "unit": "fps",
        "runtypes": ["onnxruntime-directml", "onnxruntime-tensorrt", "tensorrt", "ncnn-vulkan"],
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


def load_data(path, cols, cfg):
    """Vrati (runtypes, data, params).

    data[runtype][model] = {col: value}, params[model] = broj parametara.
    Ako je cfg.runtypes zadat, uzimaju se tacno ti runtype-ovi (tim redom).
    Inace se uzimaju svi koji pocinju sa 'onnxruntime-', poredjani po
    RUNTYPE_LABELS (nepoznati idu na kraj, azbucno).
    """
    exclude = DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude
    archs = set(cfg.archs) if cfg.archs else None
    include = cfg.include or None
    sel = set(cfg.runtypes) if cfg.runtypes else None

    data = {}
    params = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols if c not in reader.fieldnames]
        if missing:
            usable = [c for c in reader.fieldnames if c not in ("model_name", "params", "runtype")]
            sys.exit(f"Nedostaju kolone {', '.join(missing)} u {path.name}. "
                     f"Dostupne: {', '.join(usable)}")
        for r in reader:
            rt = r.get("runtype", "")
            if sel is None:
                if not rt.startswith("onnxruntime-"):
                    continue
            elif rt not in sel:
                continue
            name = r.get("model_name", "")
            if not keep(name, exclude, archs, include):
                continue
            rec = {}
            for c in cols:
                v = to_float(r.get(c))
                if v is not None:
                    rec[c] = v
            if len(rec) != len(cols):  # preskoci redove bez svih trazenih metrika
                continue
            data.setdefault(rt, {})[name] = rec
            p = to_float(r.get("params"))
            if p is not None:
                params[name] = p

    if cfg.runtypes:
        order = [rt for rt in cfg.runtypes if rt in data]
    else:
        known = [rt for rt in RUNTYPE_LABELS if rt in data]
        rest = sorted(rt for rt in data if rt not in RUNTYPE_LABELS)
        order = known + rest
    return order, data, params


def label_for(rt):
    return RUNTYPE_LABELS.get(rt, rt.replace("onnxruntime-", "ORT-"))


def plot(runtypes, group_labels, values, title, common_n, unit, legend_loc, dpi):
    """Grupisani horizontalni bar grafik: grupe na y-osi, bar po runtype-u.

    group_labels su oznake grupa (metrike ili velicine modela), a
    values[label][runtype] je vec izracunat prosek za tu grupu i runtype.
    """
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": dpi})
    is_fps = unit == "fps"
    bar_spec = " .1f" if is_fps else " .2f"
    fmt = lambda v: zf(v, bar_spec)
    xlabel = "FPS - više je bolje" if is_fps else "Vreme (ms) - niže je bolje"

    n_rt = len(runtypes)
    bar_h = 0.8 / n_rt
    fig, ax = plt.subplots(figsize=(8.4, 1.6 + 1.1 * len(group_labels)))

    centers = list(range(len(group_labels)))
    for j, rt in enumerate(runtypes):
        # y pozicija bara j unutar svake grupe, centrirano oko center-a grupe
        ys = [c + (j - (n_rt - 1) / 2) * bar_h for c in centers]
        vals = [values[g][rt] for g in group_labels]
        bars = ax.barh(ys, vals, height=bar_h, color=RUNTYPE_COLORS.get(rt, None),
                       label=label_for(rt), edgecolor="white", linewidth=0.5)
        for y, v in zip(ys, vals):
            ax.text(v, y, fmt(v), va="center", ha="left", fontsize=8.5)

    ax.set_yticks(centers)
    ax.set_yticklabels(group_labels)
    ax.invert_yaxis()  # prva grupa na vrhu
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.margins(x=0.12)
    ax.legend(loc=legend_loc, fontsize=9, framealpha=0.9)
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

    unit = cfg.unit.lower()
    if unit not in ("ms", "fps"):
        sys.exit(f"Nepoznat unit '{cfg.unit}' — koristi 'ms' ili 'fps'.")

    group_by = cfg.group_by.lower()
    if group_by not in ("metric", "size"):
        sys.exit(f"Nepoznat group_by '{cfg.group_by}' — koristi 'metric' ili 'size'.")

    cols = [f"{cfg.res} {m}" for m in cfg.metrics]
    runtypes, data, params = load_data(results_path, cols, cfg)
    if len(runtypes) < 2:
        sys.exit(f"Potrebna su bar dva backend-a; nadjeno: "
                 f"{', '.join(runtypes) or 'nijedan'}")

    # Zajednicki modeli za sve ukljucene runtype-ove.
    common = set.intersection(*(set(data[rt]) for rt in runtypes))
    if not common:
        sys.exit("Nema modela zajednickih za sve backend-ove.")
    common = sorted(common)

    # Prosek preko datog skupa modela; za fps -> prosek FPS-a po modelu (mean(1000/ms)).
    def aggregate(rt, col, models):
        vals = [data[rt][name][col] for name in models]
        if unit == "fps":
            vals = [1000.0 / v if v else 0.0 for v in vals]
        return sum(vals) / len(vals) if vals else 0.0

    # Odredi grupe (oznake na y-osi) i skup modela za svaku grupu.
    if group_by == "size":
        col = cols[0]  # samo prva metrika kod grupisanja po velicini
        groups = []  # (label, [modeli])
        for label, lo, hi in cfg.size_bins:
            members = [n for n in common if n in params
                       and params[n] >= lo and (hi is None or params[n] < hi)]
            if members:
                groups.append((label, members))
        if not groups:
            sys.exit("Nijedan zajednicki model ne upada u zadate size_bins.")
        group_labels = [label for label, _ in groups]
        values = {label: {rt: aggregate(rt, col, members) for rt in runtypes}
                  for label, members in groups}
        group_n = {label: len(members) for label, members in groups}
    else:  # group_by == "metric"
        group_labels = [METRIC_LABELS.get(m, m) for m in cfg.metrics]
        values = {METRIC_LABELS.get(m, m): {rt: aggregate(rt, col, common) for rt in runtypes}
                  for col, m in zip(cols, cfg.metrics)}
        group_n = {g: len(common) for g in group_labels}

    title = f"Runtime poredjenje - {cfg.res} ({METRIC_LABELS.get(cfg.metrics[0], cfg.metrics[0])})" \
        if group_by == "size" else f"Runtime poredjenje - {cfg.res}"

    fig = plot(runtypes, group_labels, values, title, len(common), unit,
               cfg.legend_loc, cfg.dpi)

    suffix = "_fps" if unit == "fps" else ""
    out_name = cfg.out or f"runtimes_{cfg.name}{suffix}.png"
    out_path = get_results_path("analysis/runtimes") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Izvor: {results_path}  ({cfg.res}, {unit}, group_by={group_by})")
    print(f"Backend-ovi: {', '.join(label_for(rt) for rt in runtypes)}")
    print(f"Zajednickih modela: {len(common)}")
    for g in group_labels:
        vals = "  ".join(f"{label_for(rt)}={zf(values[g][rt], '.2f')}{unit}" for rt in runtypes)
        print(f"  {g.replace(chr(10), ' '):<16} (n={group_n[g]:>2})  {vals}")
    print(f"Slika:  {out_path}")


if __name__ == "__main__":
    main()
