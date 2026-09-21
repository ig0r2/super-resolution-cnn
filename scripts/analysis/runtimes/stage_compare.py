"""
stage_compare.py - razlaganje obrade jednog frejma na faze, po backendu.

Za izabrani model crta segmentirani (stacked) horizontalni bar: jedan red po
backendu (runtype), a segmenti su trajanja faza decode / SR / display (u ms).
Duzina reda je ukupno vreme po frejmu, pa se odmah vidi i koji backend je
najbrzi i gde trosi vreme. Nazivi runtype-ova se mapiraju na citljive oznake.

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Sve konfiguracije se obradjuju u jednom pokretanju i daju odvojenu sliku
(imenovanu po "name" polju) u results/analysis/stages/.
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

# Ako je "model" jedna od ovih vrednosti, faze se uprose preko svih modela
# zajednickih za sve ukljucene backend-e (fer poredjenje na istom skupu).
AVG_KEYS = {"prosek", "average", "mean", "avg"}
# Kod proseka se ovi modeli izbacuju iz uparivanja.
DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

# Redosled i citljive oznake za runtype-ove (koristi se i kao podrazumevani
# redosled kad "runtypes" nije zadat).
RUNTYPE_LABELS = {
    "onnxruntime-cuda": "ORT-CUDA",
    "onnxruntime-directml": "ORT-DirectML",
    "onnxruntime-tensorrt": "ORT-TensorRT",
    "tensorrt": "TensorRT",
    "tensorrt-nvdec": "TensorRT+NVDEC",
    "tensorrt-nvdec-gl": "TensorRT+NVDEC+GL",
    "ncnn-vulkan": "ncnn-Vulkan",
    "opengl": "OpenGL",
}

# Faze (kratka oznaka -> deo imena kolone, spaja se sa "res": f"{res} {key}").
STAGE_LABELS = {
    "decode": "Dekodovanje",
    "sr": "SR",
    "display": "Prikaz",
}
STAGE_COLORS = {
    "decode": "#3c9a5f",
    "sr": "#1b6ca8",
    "display": "#e8702a",
}

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "default",  # koristi se za ime izlazne slike
    "results": "results_2x_video_ms.csv",  # CSV sa video merenjem (ms)
    "res": "480p",  # prefiks rezolucije u imenu kolone
    "model": "SR_EDSR_2_52",  # tacno ime modela, ili "prosek" (prosek preko
    # zajednickih modela svih ukljucenih backend-a)
    "stages": ["decode", "sr", "display"],  # faze i njihov redosled u baru
    "runtypes": None,  # None -> svi backend-i koji imaju taj model
    # (redom iz RUNTYPE_LABELS); ili lista tacnih
    "out": None,  # None -> stages_<name>.png
    "legend_loc": "lower right",  # pozicija legende (matplotlib loc)
    "dpi": 140,
}

##############################################


CONFIGS = [
    {
        "name": "prosek",
        "model": "prosek",
    },
    {
        "name": "prosek_trt",
        "model": "prosek",
        "runtypes": ["onnxruntime-tensorrt", "tensorrt", "tensorrt-nvdec", "tensorrt-nvdec-gl"],
    },
    {
        "name": "prosek_decode",
        "model": "prosek",
        "runtypes": ["tensorrt", "tensorrt-nvdec"]
    },
    {
        "name": "prosek_display",
        "model": "prosek",
        "runtypes": ["tensorrt", "tensorrt-nvdec", "tensorrt-nvdec-gl"]
    },
    {
        "name": "FastEDSR_2_32_decode",
        "model": "SR_FastEDSR_2_32",
        "runtypes": ["tensorrt", "tensorrt-nvdec"]
    },
    {
        "name": "FastEDSR_2_32_display",
        "model": "SR_FastEDSR_2_32",
        "runtypes": ["tensorrt", "tensorrt-nvdec", "tensorrt-nvdec-gl"]
    },
    {
        "name": "FastEDSR_4_128_decode",
        "model": "SR_FastEDSR_4_128",
        "runtypes": ["tensorrt", "tensorrt-nvdec"]
    },
    {
        "name": "FastEDSR_4_128_display",
        "model": "SR_FastEDSR_4_128",
        "runtypes": ["tensorrt", "tensorrt-nvdec", "tensorrt-nvdec-gl"]
    },
]


##############################################


def resolve_results(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return get_results_path(name)


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def zf(value, spec) -> str:
    """Formatira broj po `spec` sa decimalnim zarezom umesto tacke."""
    return format(value, spec).replace(".", ",")


def label_for(rt):
    return RUNTYPE_LABELS.get(rt, rt)


def load_model(path, cols, cfg):
    """Vrati (runtypes, data, n_models); data[runtype] = {col: value}.

    Ako je cfg.model iz AVG_KEYS, faze su prosek preko modela zajednickih za sve
    ukljucene backend-e (n_models = broj tih modela); inace je to jedan model
    (n_models = 1).
    """
    sel = set(cfg.runtypes) if cfg.runtypes else None
    is_avg = str(cfg.model).strip().lower() in AVG_KEYS
    per_rt = {}  # rt -> {model_name: {col: value}} (samo redovi sa svim fazama)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols if c not in reader.fieldnames]
        if missing:
            usable = [c for c in reader.fieldnames if c not in ("model_name", "params", "runtype")]
            sys.exit(f"Nedostaju kolone {', '.join(missing)} u {path.name}. "
                     f"Dostupne: {', '.join(usable)}")
        for r in reader:
            name = r.get("model_name", "")
            if not is_avg and name != cfg.model:
                continue
            if is_avg and any(sub in name for sub in DEFAULT_EXCLUDE):
                continue
            rt = r.get("runtype", "")
            if sel is not None and rt not in sel:
                continue
            rec = {}
            for c in cols:
                v = to_float(r.get(c))
                if v is not None:
                    rec[c] = v
            if len(rec) != len(cols):  # preskoci redove bez svih faza
                continue
            per_rt.setdefault(rt, {})[name] = rec

    if is_avg:
        # zajednicki modeli za sve ukljucene backend-e, pa prosek po fazi.
        rts = list(per_rt)
        common = set.intersection(*(set(per_rt[rt]) for rt in rts)) if rts else set()
        if not common:
            data, n_models = {}, 0
        else:
            data = {rt: {c: sum(per_rt[rt][m][c] for m in common) / len(common)
                         for c in cols} for rt in rts}
            n_models = len(common)
    else:
        data = {rt: recs[cfg.model] for rt, recs in per_rt.items()
                if cfg.model in recs}
        n_models = 1 if data else 0

    if cfg.runtypes:
        order = [rt for rt in cfg.runtypes if rt in data]
    else:
        known = [rt for rt in RUNTYPE_LABELS if rt in data]
        rest = sorted(rt for rt in data if rt not in RUNTYPE_LABELS)
        order = known + rest
    return order, data, n_models


def plot(runtypes, data, stages, cols, title, res, legend_loc, dpi):
    """Stacked horizontalni bar: red = backend, segmenti = faze (ms)."""
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": dpi})
    fig, ax = plt.subplots(figsize=(9.0, 1.4 + 0.55 * len(runtypes)))

    ys = list(range(len(runtypes)))
    # prag ispod kog se ne crta tekst u segmentu (preusko -> secenje/preklapanje)
    max_total = max(sum(data[rt][c] for c in cols) for rt in runtypes)
    min_w = 0.05 * max_total
    for stage, col in zip(stages, cols):
        lefts = []
        widths = []
        for rt in runtypes:
            # kumulativni pocetak = zbir prethodnih faza za taj backend
            prev = sum(data[rt][c] for s, c in zip(stages, cols)
                       if stages.index(s) < stages.index(stage))
            lefts.append(prev)
            widths.append(data[rt][col])
        ax.barh(ys, widths, left=lefts, height=0.62,
                color=STAGE_COLORS.get(stage), label=STAGE_LABELS.get(stage, stage),
                edgecolor="white", linewidth=0.5)
        # oznaka vrednosti u sredini segmenta ako je dovoljno sirok
        for y, l, w in zip(ys, lefts, widths):
            if w >= min_w:
                ax.text(l + w / 2, y, zf(w, ".2f"), va="center", ha="center",
                        fontsize=7.5, color="white")

    # ukupno vreme na kraju reda
    for y, rt in zip(ys, runtypes):
        total = sum(data[rt][c] for c in cols)
        ax.text(total, y, f" {zf(total, '.2f')}", va="center", ha="left", fontsize=8.5)

    ax.set_yticks(ys)
    ax.set_yticklabels([label_for(rt) for rt in runtypes])
    ax.invert_yaxis()  # prvi backend na vrhu
    ax.set_xlabel("Vreme po frejmu (ms) - nize je bolje")
    ax.set_title(f"Faze obrade frejma - {title} @ {res}")
    ax.margins(x=0.10)
    ax.legend(loc=legend_loc, fontsize=9, framealpha=0.9, ncol=len(stages))
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

    cols = [f"{cfg.res} {s}" for s in cfg.stages]
    runtypes, data, n_models = load_model(results_path, cols, cfg)
    if not runtypes:
        sys.exit(f"Model '{cfg.model}' nema kompletne faze ni za jedan backend "
                 f"(proveri model/res/runtypes).")

    is_avg = str(cfg.model).strip().lower() in AVG_KEYS
    title = f"prosek ({n_models} modela)" if is_avg else cfg.model

    fig = plot(runtypes, data, cfg.stages, cols, title, cfg.res,
               cfg.legend_loc, cfg.dpi)

    out_name = cfg.out or f"stages_{cfg.name}.png"
    out_path = get_results_path("analysis/stages") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    model_info = f"prosek/{n_models} modela" if is_avg else cfg.model
    print(f"Izvor: {results_path}  ({cfg.res}, model={model_info})")
    print(f"Backend-ovi: {', '.join(label_for(rt) for rt in runtypes)}")
    for rt in runtypes:
        parts = "  ".join(f"{STAGE_LABELS.get(s, s)}={zf(data[rt][c], '.2f')}"
                          for s, c in zip(cfg.stages, cols))
        total = sum(data[rt][c] for c in cols)
        print(f"  {label_for(rt):<18} {parts}   Ukupno={zf(total, '.2f')}ms")
    print(f"Slika:  {out_path}")


if __name__ == "__main__":
    main()
