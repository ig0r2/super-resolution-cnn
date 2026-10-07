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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.plotting import save_fig, setup_style
import matplotlib.pyplot as plt

from utils.analysis.config import run_configs
from utils.analysis.data import name_filter, read_rows, require_results, to_float
from utils.analysis.fmt import zf
from utils.analysis.runtypes import order_runtypes, runtype_label
from utils.path import get_results_path

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

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
    "size_bins": [  # koristi se samo kad je group_by == "size" (prva metrika)
        ("Mali\nmodeli", 0, 100_000),  # params u [0, 100k)
        ("Veliki\nmodeli", 100_000, None),  # params >= 100k (None = bez gornje granice)
    ],
    "archs": None,  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,  # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,  # None -> DEFAULT_EXCLUDE
    "out": None,  # None -> runtimes_<name>.png
    "legend_loc": "upper left",  # pozicija legende (matplotlib loc), "upper left", "lower right", "best"
    "show_title": False,  # True -> naslov iznad grafika
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "ort_sr",
        "metrics": ["sr"],
        "unit": "fps",
        "runtypes": ["onnxruntime-cuda", "onnxruntime-directml", "onnxruntime-tensorrt"],
    },
    {
        "name": "ort_sr_by_size",
        "metrics": ["sr"],
        "unit": "fps",
        "group_by": "size",
        "runtypes": ["onnxruntime-cuda", "onnxruntime-directml", "onnxruntime-tensorrt"],
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
        "name": "ncnn",
        "metrics": ["sr"],
        "unit": "fps",
        "runtypes": ["onnxruntime-directml", "ncnn-vulkan"],
        "legend_loc": "upper left"
    },
    {
        "name": "ncnn_by_size",
        "metrics": ["sr"],
        "unit": "fps",
        "group_by": "size",
        "runtypes": ["onnxruntime-directml", "ncnn-vulkan"],
        "legend_loc": "lower right"
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


def load_data(path, cols, cfg):
    """Vrati (runtypes, data, params).

    data[runtype][model] = {col: value}, params[model] = broj parametara.
    Ako je cfg.runtypes zadat, uzimaju se tacno ti runtype-ovi (tim redom).
    Inace se uzimaju svi koji pocinju sa 'onnxruntime-', poredjani po
    RUNTYPE_LABELS (nepoznati idu na kraj, azbucno).
    """
    keep = name_filter(DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude, cfg.archs, cfg.include)
    sel = set(cfg.runtypes) if cfg.runtypes else None

    data = {}
    params = {}
    for r in read_rows(path, cols):
        rt = r.get("runtype", "")
        if sel is None:
            if not rt.startswith("onnxruntime-"):
                continue
        elif rt not in sel:
            continue
        name = r.get("model_name", "")
        if not keep(name):
            continue
        rec = {c: to_float(r.get(c)) for c in cols}
        if any(v is None for v in rec.values()):  # preskoci redove bez svih trazenih metrika
            continue
        data.setdefault(rt, {})[name] = rec
        p = to_float(r.get("params"))
        if p is not None:
            params[name] = p

    return order_runtypes(data, cfg.runtypes), data, params


def plot(runtypes, group_labels, values, title, unit, legend_loc, dpi, show_title=False):
    """Grupisani horizontalni bar grafik: grupe na y-osi, bar po runtype-u.

    group_labels su oznake grupa (metrike ili velicine modela), a
    values[label][runtype] je vec izracunat prosek za tu grupu i runtype.
    """
    setup_style(dpi)
    is_fps = unit == "fps"
    bar_spec = " .1f" if is_fps else " .2f"
    xlabel = "FPS - više je bolje" if is_fps else "Vreme (ms) - niže je bolje"

    n_rt = len(runtypes)
    bar_h = 0.8 / n_rt
    fig, ax = plt.subplots(figsize=(8.4, 1.6 + 1.1 * len(group_labels)))

    centers = list(range(len(group_labels)))
    for j, rt in enumerate(runtypes):
        # y pozicija bara j unutar svake grupe, centrirano oko center-a grupe
        ys = [c + (j - (n_rt - 1) / 2) * bar_h for c in centers]
        vals = [values[g][rt] for g in group_labels]
        ax.barh(ys, vals, height=bar_h, color=RUNTYPE_COLORS.get(rt, None),
                label=runtype_label(rt), edgecolor="white", linewidth=0.5)
        for y, v in zip(ys, vals):
            ax.text(v, y, zf(v, bar_spec), va="center", ha="left", fontsize=8.5)

    ax.set_yticks(centers)
    ax.set_yticklabels(group_labels)
    ax.invert_yaxis()  # prva grupa na vrhu
    ax.set_xlabel(xlabel)
    if show_title:
        ax.set_title(title)
    ax.margins(x=0.12)
    ax.legend(loc=legend_loc, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def run_config(cfg):
    results_path = require_results(cfg.results)

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

    fig = plot(runtypes, group_labels, values, title, unit, cfg.legend_loc, cfg.dpi, cfg.show_title)

    suffix = "_fps" if unit == "fps" else ""
    out_path = save_fig(fig, get_results_path("analysis/runtimes")
                        / (cfg.out or f"runtimes_{cfg.name}{suffix}.png"))

    print(f"Izvor: {results_path}  ({cfg.res}, {unit}, group_by={group_by})")
    print(f"Backend-ovi: {', '.join(runtype_label(rt) for rt in runtypes)}")
    print(f"Zajednickih modela: {len(common)}")
    for g in group_labels:
        vals = "  ".join(f"{runtype_label(rt)}={zf(values[g][rt], '.2f')}{unit}" for rt in runtypes)
        print(f"  {g.replace(chr(10), ' '):<16} (n={group_n[g]:>2})  {vals}")
    print(f"Slika:  {out_path}")


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
