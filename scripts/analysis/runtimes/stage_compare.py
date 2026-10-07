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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.plotting import save_fig, setup_style
import matplotlib.pyplot as plt

from utils.analysis.config import run_configs
from utils.analysis.data import read_rows, require_results, to_float
from utils.analysis.fmt import zf
from utils.analysis.names import display_name
from utils.analysis.runtypes import order_runtypes, runtype_label
from utils.path import get_results_path

# Ako je "model" jedna od ovih vrednosti, faze se uprose preko svih modela
# zajednickih za sve ukljucene backend-e (fer poredjenje na istom skupu).
AVG_KEYS = {"prosek", "average", "mean", "avg"}
# Kod proseka se ovi modeli izbacuju iz uparivanja.
DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]

# Faze (kratka oznaka -> deo imena kolone, spaja se sa "res": f"{res} {key}").
STAGE_LABELS = {
    "decode": "Dekodiranje",
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
    # (redom iz RUNTYPE_LABELS); ili lista tacnih runtype-ova; ili dict
    # {runtype: oznaka} - zadata oznaka zamenjuje onu iz RUNTYPE_LABELS
    # (None -> oznaka iz RUNTYPE_LABELS)
    "out": None,  # None -> stages_<name>.png
    "legend_loc": "lower right",  # pozicija legende (matplotlib loc)
    "show_title": False,  # True -> naslov iznad grafika
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
        "runtypes": {"tensorrt": "cv2 dekodiranje", "tensorrt-nvdec": "NVDEC dekodiranje"},
    },
    {
        "name": "prosek_display",
        "model": "prosek",
        "runtypes": {"tensorrt": "cv2 dekodiranje\ncv2 prikaz", "tensorrt-nvdec": "NVDEC dekodiranje\ncv2 prikaz",
                     "tensorrt-nvdec-gl": "NVDEC dekodiranje\nOpenGL prikaz"},
    },
    {
        "name": "FastEDSR_4_32",
        "model": "SR_FastEDSR_4_32",
        "runtypes": {"tensorrt": "cv2 dekodiranje\ncv2 prikaz", "tensorrt-nvdec-gl": "NVDEC dekodiranje\nOpenGL prikaz"}
    },
    {
        "name": "FastEDSR_4_32_3",
        "model": "SR_FastEDSR_4_32",
        "runtypes": {"tensorrt": "cv2 dekodiranje\ncv2 prikaz", "tensorrt-nvdec": "NVDEC dekodiranje\ncv2 prikaz",
                     "tensorrt-nvdec-gl": "NVDEC dekodiranje\nOpenGL prikaz"},
    },
    {
        "name": "FastEDSR_4_32_shaders",
        "model": "SR_FastEDSR_4_32",
        "runtypes": {"tensorrt": "cv2 dekodiranje\ncv2 prikaz",
                     "tensorrt-nvdec-gl": "NVDEC dekodiranje\nOpenGL prikaz",
                     "opengl": "NVDEC dekodiranje\nOpenGL šejderi"}
    },
    {
        "name": "FastEDSR_4_128",
        "model": "SR_FastEDSR_4_128",
        "runtypes": {"tensorrt": "cv2 dekodiranje\ncv2 prikaz", "tensorrt-nvdec-gl": "NVDEC dekodiranje\nOpenGL prikaz"}
    },
]


##############################################


def normalize_runtypes(runtypes):
    """Vrati (lista runtype-ova ili None, {runtype: oznaka}) iz cfg.runtypes.

    Prihvata None, listu runtype-ova ili dict {runtype: oznaka}.
    """
    if not runtypes:
        return None, {}
    if isinstance(runtypes, dict):
        return list(runtypes), {rt: lbl for rt, lbl in runtypes.items() if lbl}
    return list(runtypes), {}


def load_model(path, cols, cfg):
    """Vrati (runtypes, data, n_models); data[runtype] = {col: value}.

    Ako je cfg.model iz AVG_KEYS, faze su prosek preko modela zajednickih za sve
    ukljucene backend-e (n_models = broj tih modela); inace je to jedan model
    (n_models = 1).
    """
    sel = set(cfg.runtypes) if cfg.runtypes else None
    is_avg = str(cfg.model).strip().lower() in AVG_KEYS
    per_rt = {}  # rt -> {model_name: {col: value}} (samo redovi sa svim fazama)
    for r in read_rows(path, cols):
        name = r.get("model_name", "")
        if not is_avg and name != cfg.model:
            continue
        if is_avg and any(sub in name for sub in DEFAULT_EXCLUDE):
            continue
        rt = r.get("runtype", "")
        if sel is not None and rt not in sel:
            continue
        rec = {c: to_float(r.get(c)) for c in cols}
        if any(v is None for v in rec.values()):  # preskoci redove bez svih faza
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

    return order_runtypes(data, cfg.runtypes), data, n_models


def plot(runtypes, data, stages, cols, title, res, legend_loc, dpi, labels=None,
         show_title=False):
    """Stacked horizontalni bar: red = backend, segmenti = faze (ms)."""
    setup_style(dpi)
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
    ax.set_yticklabels([runtype_label(rt, labels) for rt in runtypes])
    ax.invert_yaxis()  # prvi backend na vrhu
    ax.set_xlabel("Vreme po frejmu (ms) - nize je bolje")
    if show_title:
        ax.set_title(f"Faze obrade frejma - {title} @ {res}")
    ax.margins(x=0.10)
    ax.legend(loc=legend_loc, fontsize=9, framealpha=0.9, ncol=len(stages))
    fig.tight_layout()
    return fig


def run_config(cfg):
    results_path = require_results(cfg.results)

    cfg.runtypes, labels = normalize_runtypes(cfg.runtypes)
    cols = [f"{cfg.res} {s}" for s in cfg.stages]
    runtypes, data, n_models = load_model(results_path, cols, cfg)
    if not runtypes:
        sys.exit(f"Model '{cfg.model}' nema kompletne faze ni za jedan backend "
                 f"(proveri model/res/runtypes).")

    is_avg = str(cfg.model).strip().lower() in AVG_KEYS
    title = f"prosek ({n_models} modela)" if is_avg else display_name(cfg.model)

    fig = plot(runtypes, data, cfg.stages, cols, title, cfg.res,
               cfg.legend_loc, cfg.dpi, labels, cfg.show_title)

    out_path = save_fig(fig, get_results_path("analysis/stages") / (cfg.out or f"stages_{cfg.name}.png"))

    model_info = f"prosek/{n_models} modela" if is_avg else display_name(cfg.model)
    print(f"Izvor: {results_path}  ({cfg.res}, model={model_info})")
    print(f"Backend-ovi: {', '.join(runtype_label(rt, labels) for rt in runtypes)}")
    for rt in runtypes:
        parts = "  ".join(f"{STAGE_LABELS.get(s, s)}={zf(data[rt][c], '.2f')}"
                          for s, c in zip(cfg.stages, cols))
        total = sum(data[rt][c] for c in cols)
        print(f"  {runtype_label(rt, labels):<18} {parts}   Ukupno={zf(total, '.2f')}ms")
    print(f"Slika:  {out_path}")


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
