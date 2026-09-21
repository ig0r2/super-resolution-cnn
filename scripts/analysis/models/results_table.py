"""
results_table.py - LaTeX tabela najboljih modela po klasi (arhitekturi).

Za zadati scale i dataset (config), iz odgovarajuceg results CSV-a bira najbolji
model iz svake klase arhitektura (po primarnoj metrici, podrazumevano PSNR) i
ispisuje gotovu LaTeX (booktabs) tabelu: kolone Parametri (10^6), PSNR, SSIM,
LPIPS; bolduje se najbolja vrednost po koloni. Opciono se na vrh dodaju bazne
interpolacije (bicubic/lanczos) bez broja parametara.
"""

import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path import get_results_path

# Konzola je cesto cp1252; caption sadrzi ć/ž pa reconfigure da print ne puca.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Metrika -> (broj decimala, vece_je_bolje, strelica, jedinica ili "").
METRIC_INFO = {
    "PSNR": (4, True, r"$\uparrow$", "dB"),
    "SSIM": (4, True, r"$\uparrow$", ""),
    "LPIPS": (4, False, r"$\downarrow$", ""),
}

# Citljive oznake baznih interpolacija (mala slova u CSV-u -> prikaz).
BASELINE_LABELS = {
    "nearest": "Nearest", "bilinear": "Bilinear",
    "bicubic": "Bicubic", "lanczos": "Lanczos",
}

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "standard",             # koristi se za ime .tex fajla
    "scale": "2x",                  # uvecanje (2x/3x/4x) — ulazi u ime CSV-a
    "dataset": "DIV2K",             # skup — ulazi u ime CSV-a
    "results": None,                # None -> results_{scale}_{dataset}_half.csv
    "models": [],                   # eksplicitna lista model_name (tim redom);
                                    # [] -> auto: najbolji po klasi iz "classes"
    "classes": ["SRCNN", "VDSR", "EDSR", "IMDN", "RFDN"],  # klase, tim redom
    "baselines": ["nearest", "bilinear", "bicubic", "lanczos"],  # bazne interpolacije na vrhu ([] za bez)
    "primary": "PSNR",              # metrika po kojoj se bira najbolji u klasi
    "metrics": ["PSNR", "SSIM", "LPIPS"],  # kolone metrika (redom)
    "decimals": None,               # decimale metrika: None -> podrazumevano (4);
                                    # int za sve; ili dict {"PSNR": 2, ...}
    "param_decimals": 3,            # decimale za kolonu parametara (10^6)
    "caption": None,                # None -> automatski
    "label": None,                  # None -> tab:results_{scale}_{dataset}
    "out": None,                    # None -> results/analysis/models/results_table_<name>.tex
}

##############################################


CONFIGS = [
    {
        "name": "standard_2x_DIV2K",
        "scale": "2x",
        "dataset": "DIV2K",
    },
    {
        "name": "standard_2x_Set14",
        "scale": "2x",
        "dataset": "Set14",
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


def arch_of(name: str) -> str:
    parts = name.split("_")
    if parts and parts[0] == "SR":
        parts = parts[1:]
    return parts[0] if parts else name


def display_name(name: str) -> str:
    """SR_EDSR_2x_32_256_r -> 'EDSR 32/256$_{r}$'."""
    parts = name.split("_")
    if parts and parts[0] == "SR":
        parts = parts[1:]
    arch, rest = parts[0], parts[1:]
    rest = [p for p in rest if not re.fullmatch(r"\d+x", p)]  # izbaci scale token
    nums = [p for p in rest if p.isdigit()]
    subs = [p for p in rest if not p.isdigit()]
    label = arch
    if nums:
        label += " " + "/".join(nums)
    for s in subs:
        label += f"$_{{{s}}}$"
    return label


def load_rows(path, metrics):
    """Vrati {model_name: {"params": p, metric: value, ...}} sa svim metrikama."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [m for m in metrics if m not in reader.fieldnames]
        if missing:
            sys.exit(f"Metrike {', '.join(missing)} ne postoje u {path.name}.")
        for r in reader:
            name = r.get("model_name", "")
            if not name:
                continue
            rec = {"params": to_float(r.get("params"))}
            ok = True
            for m in metrics:
                v = to_float(r.get(m))
                if v is None:
                    ok = False
                    break
                rec[m] = v
            if ok:
                out[name] = rec
    return out


def best_per_class(rows, cfg):
    """Za svaku klasu iz cfg.classes vrati najbolji model (po cfg.primary)."""
    _, higher_better, _, _ = METRIC_INFO[cfg.primary]
    scale_tok = cfg.scale
    picked = {}  # arch -> (name, rec)
    for name, rec in rows.items():
        if any(sub in name for sub in ("GAN", "ESRGAN", "jpeg")):
            continue
        parts = name.split("_")
        if parts[0] != "SR" or scale_tok not in parts:
            continue
        arch = arch_of(name)
        if arch not in cfg.classes:
            continue
        cur = picked.get(arch)
        better = cur is None or (
            rec[cfg.primary] > cur[1][cfg.primary] if higher_better
            else rec[cfg.primary] < cur[1][cfg.primary])
        if better:
            picked[arch] = (name, rec)
    # zadrzi redosled iz cfg.classes
    return [picked[a] for a in cfg.classes if a in picked]


def main():
    if not CONFIGS:
        sys.exit("CONFIGS je prazna - dodaj bar jednu konfiguraciju.")
    for i, cfg_dict in enumerate(CONFIGS):
        cfg = SimpleNamespace(**{**CONFIG_DEFAULTS, **cfg_dict})
        if i:
            print("\n" + "=" * 96 + "\n")
        print(f"### Konfiguracija: {cfg.name}")
        run_config(cfg)


def run_config(cfg):
    for m in cfg.metrics + [cfg.primary]:
        if m not in METRIC_INFO:
            sys.exit(f"Nepoznata metrika '{m}'. Podrzane: {', '.join(METRIC_INFO)}")

    results_name = cfg.results or f"results_{cfg.scale}_{cfg.dataset}_half.csv"
    results_path = resolve_results(results_name)
    if not results_path.exists():
        sys.exit(f"CSV ne postoji: {results_path}")

    rows = load_rows(results_path, cfg.metrics)
    if cfg.models:
        # Eksplicitno zadati modeli, tim redom.
        models = []
        for name in cfg.models:
            if name in rows:
                models.append((name, rows[name]))
            else:
                print(f"  [upozorenje] model '{name}' nije nadjen u {results_path.name} — preskacem")
        if not models:
            sys.exit("Nijedan od zadatih 'models' nije nadjen u CSV-u.")
    else:
        models = best_per_class(rows, cfg)
        if not models:
            sys.exit(f"Nijedna klasa ({', '.join(cfg.classes)}) nema model sa "
                     f"'{cfg.scale}' tokenom u {results_path.name}.")

    baselines = [(BASELINE_LABELS.get(b, b.capitalize()), rows[b])
                 for b in cfg.baselines if b in rows]

    tex = tex_table(cfg, models, baselines)
    print(tex)

    out_path = Path(cfg.out) if cfg.out else get_results_path(
        f"analysis/models/results_table_{cfg.name}.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex, encoding="utf-8")
    print(f"\nLaTeX tabela: {out_path}")


def tex_table(cfg, models, baselines):
    scale_num = cfg.scale.rstrip("x")
    caption = cfg.caption or (
        f"Kvalitet izdvojenih standardnih arhitektura na skupu {cfg.dataset} za "
        f"uvećanje $\\times{scale_num}$. Broj parametara izražen je u milionima i "
        f"zaokružen na tri decimale. Strelice označavaju poželjan smer promene "
        f"metrike.")
    label = cfg.label or f"tab:results_{cfg.scale}_{cfg.dataset}"

    # Najbolja vrednost po metrici (samo medju modelima, ne baznim).
    best = {}
    for m in cfg.metrics:
        _, higher_better, _, _ = METRIC_INFO[m]
        vals = [rec[m] for _, rec in models]
        best[m] = max(vals) if higher_better else min(vals)

    def prec_for(m):
        d = cfg.decimals
        if d is None:
            return METRIC_INFO[m][0]
        return d.get(m, METRIC_INFO[m][0]) if isinstance(d, dict) else d

    def cell(rec, m, bold=True):
        s = zf(rec[m], f".{prec_for(m)}f")
        if bold and rec[m] == best[m]:
            s = r"\textbf{" + s + "}"
        return s

    # Zaglavlje.
    headers = [r"\shortstack{Parametri\\($10^6$)}"]
    for m in cfg.metrics:
        _, _, arrow, unit = METRIC_INFO[m]
        if unit:
            headers.append(r"\shortstack{" + f"{m} {arrow}" + r"\\(" + unit + ")}")
        else:
            headers.append(f"{m} {arrow}")

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{{caption}}}")
    lines.append(f"    \\label{{{label}}}")
    lines.append(r"    \begingroup")
    lines.append(r"    \small")
    lines.append(r"    \setlength{\tabcolsep}{4pt}")
    lines.append(r"    \renewcommand{\arraystretch}{1.15}")
    lines.append(r"    \begin{tabular}{@{}l" + "r" * (1 + len(cfg.metrics)) + "@{}}")
    lines.append(r"        \toprule")
    lines.append("        Model & " + " & ".join(headers) + r" \\")
    lines.append(r"        \midrule")

    for disp, rec in baselines:
        cells = [disp, "---"] + [cell(rec, m, bold=False) for m in cfg.metrics]
        lines.append("        " + " & ".join(cells) + r" \\")
    if baselines:
        lines.append(r"        \midrule")

    for name, rec in models:
        params = zf(rec['params'] / 1e6, f".{cfg.param_decimals}f") if rec.get("params") else "---"
        cells = [display_name(name), params] + [cell(rec, m) for m in cfg.metrics]
        lines.append("        " + " & ".join(cells) + r" \\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \endgroup")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
