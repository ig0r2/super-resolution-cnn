"""
results_table.py - LaTeX tabela najboljih modela po klasi (arhitekturi).

Za zadati scale i dataset (config), iz odgovarajuceg results CSV-a bira najbolji
model iz svake klase arhitektura (po primarnoj metrici, podrazumevano PSNR) i
ispisuje gotovu LaTeX (booktabs) tabelu: kolone Parametri (10^6), PSNR, SSIM,
LPIPS, uz sive Δ kolone (razlika prema bikubnoj interpolaciji). Opciono se na
vrh dodaju bazne interpolacije bez broja parametara, bolduje najbolja vrednost
po koloni i izostavljaju konfiguracija modela (B/n_f) i kolona parametara.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.config import run_configs
from utils.analysis.data import read_rows, require_results, results_csv, to_float
from utils.analysis.fmt import tex_row, tex_table, zf
from utils.analysis.metrics import METRICS, check_metrics, higher_is_better, prec_for
from utils.analysis.names import arch_of, is_multiscale, is_scale_token
from utils.path import get_results_path

# Citljive oznake baznih interpolacija (mala slova u CSV-u -> prikaz).
BASELINE_LABELS = {
    "nearest": "Nearest", "bilinear": "Bilinear",
    "bicubic": "Bicubic", "lanczos": "Lanczos",
}

# delta_ref -> oznaka u caption-u ("razlika u odnosu na ...").
DELTA_REF_CAPTION = {
    "nearest": "interpolaciju najbližim susedom", "bilinear": "bilinearnu interpolaciju",
    "bicubic": "bikubnu interpolaciju", "lanczos": "Lanczos interpolaciju",
}

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "standard",  # koristi se za ime .tex fajla
    "scale": "2x",  # uvecanje (2x/3x/4x) — ulazi u ime CSV-a
    "dataset": "DIV2K",  # skup — ulazi u ime CSV-a
    "results": None,  # None -> results_{scale}_{dataset}_half.csv
    "models": [],  # eksplicitna lista model_name (tim redom);
    # [] -> auto: najbolji po klasi iz "classes"
    "classes": ["SRCNN", "VDSR", "EDSR", "IMDN", "RFDN", "ABPN", "FastEDSR"],  # klase, tim redom
    "include_multiscale": True,  # True -> u izbor ulaze i multiscale modeli
    "baselines": ["nearest", "bilinear", "bicubic", "lanczos"],  # bazne interpolacije na vrhu ([] za bez)
    "primary": "PSNR",  # metrika po kojoj se bira najbolji u klasi
    "metrics": ["PSNR", "SSIM", "LPIPS"],  # kolone metrika (redom)
    "decimals": {"PSNR": 2, "SSIM": 4, "LPIPS": 4},  # broj decimala: None -> podrazumevano (4); int za sve;
    # ili dict npr. {"PSNR": 2, "SSIM": 4, "LPIPS": 3}
    "param_decimals": 3,  # decimale za kolonu parametara (10^6)
    "deltas": True,  # True -> posle svake metrike siva kolona Δ u odnosu na delta_ref
    "delta_ref": "bicubic",  # bazna interpolacija (model_name iz CSV-a) prema kojoj se racuna Δ
    "bold": False,  # True -> bolduje najbolju vrednost po koloni (medju modelima)
    "show_config": True,  # False -> samo ime arhitekture, bez B/n_f
    "show_params": True,  # False -> bez kolone Parametri
    "header_units": True,  # False -> bez "(dB)" reda u zaglavlju
    "metric_gap": "",  # razmak ispred zaglavlja svake metrike osim prve (npr. r"\qquad")
    "caption": None,  # None -> automatski
    "label": None,  # None -> tab:results_{scale}_{dataset}
    "out": None,  # None -> results/analysis/models/results_table_<name>.tex
}

##############################################


CONFIGS = [
    {
        "name": "2x_DIV2K",
        "scale": "2x",
        "dataset": "DIV2K",
    },
    {
        "name": "2x_DIV2K_short",
        "scale": "2x",
        "dataset": "DIV2K",
        "show_config": False,
        "show_params": False,
        "header_units": False,
        "metric_gap": r"\qquad",
        "label": "tab:results_2x_DIV2K_short",
    },
    {
        "name": "4x_DIV2K",
        "scale": "4x",
        "dataset": "DIV2K",
    },
]


##############################################


def display_name(name: str, with_config: bool = True) -> str:
    """SR_EDSR_2x_32_256_r -> 'EDSR 32/256' (_r se ignorise); nenumericki tokeni idu
    u indeks ('FastEDSR 4/64$_{NN}$'); bez konfiguracije -> 'EDSR'."""
    parts = name.split("_")
    if parts and parts[0] == "SR":
        parts = parts[1:]
    arch, rest = parts[0], parts[1:]
    rest = [p for p in rest if not is_scale_token(p) and p != "r"]  # izbaci scale token i _r
    nums = [p for p in rest if p.isdigit()]
    subs = [p for p in rest if not p.isdigit()]
    label = arch
    if nums and with_config:
        label += " " + "/".join(nums)
    for s in subs:
        label += f"$_{{{s}}}$"
    return label


def load_rows(path, metrics):
    """Vrati {model_name: {"params": p, metric: value, ...}} sa svim metrikama."""
    out = {}
    for r in read_rows(path, metrics):
        name = r.get("model_name", "")
        if not name:
            continue
        rec = {"params": to_float(r.get("params"))}
        rec.update((m, to_float(r.get(m))) for m in metrics)
        if all(rec[m] is not None for m in metrics):
            out[name] = rec
    return out


def best_per_class(rows, cfg):
    """Za svaku klasu iz cfg.classes vrati najbolji model (po cfg.primary)."""
    higher = higher_is_better(cfg.primary)
    picked = {}  # arch -> (name, rec)
    for name, rec in rows.items():
        if any(sub in name for sub in ("GAN", "ESRGAN", "jpeg")):
            continue
        if not name.startswith("SR_"):
            continue
        if cfg.scale not in name.split("_"):
            if not (cfg.include_multiscale and is_multiscale(name)):
                continue
        arch = arch_of(name)
        if arch not in cfg.classes:
            continue
        cur = picked.get(arch)
        better = cur is None or (
            rec[cfg.primary] > cur[1][cfg.primary] if higher
            else rec[cfg.primary] < cur[1][cfg.primary])
        if better:
            picked[arch] = (name, rec)
    # zadrzi redosled iz cfg.classes
    return [picked[a] for a in cfg.classes if a in picked]


def run_config(cfg):
    check_metrics(cfg.metrics + [cfg.primary])
    results_path = require_results(cfg.results or results_csv(cfg.scale, cfg.dataset))

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
            extra = " (ni multiscale)" if cfg.include_multiscale else ""
            sys.exit(f"Nijedna klasa ({', '.join(cfg.classes)}) nema model sa "
                     f"'{cfg.scale}' tokenom{extra} u {results_path.name}.")

    baselines = [(BASELINE_LABELS.get(b, b.capitalize()), rows[b])
                 for b in cfg.baselines if b in rows]

    ref = None
    if cfg.deltas:
        ref = rows.get(cfg.delta_ref)
        if ref is None:
            print(f"  [upozorenje] delta_ref '{cfg.delta_ref}' nije nadjen u "
                  f"{results_path.name} — tabela bez Δ kolona")

    tex = build_table(cfg, models, baselines, ref)
    print(tex)

    out_path = Path(cfg.out) if cfg.out else get_results_path(
        f"analysis/models/results_table_{cfg.name}.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex, encoding="utf-8")
    print(f"\nLaTeX tabela: {out_path}")


def build_table(cfg, models, baselines, ref=None):
    scale_num = cfg.scale.rstrip("x")
    ref_label = DELTA_REF_CAPTION.get(cfg.delta_ref, cfg.delta_ref)
    caption = cfg.caption or (
        f"Kvalitet izdvojenih standardnih arhitektura na skupu {cfg.dataset} za "
        f"uvećanje $\\times{scale_num}$."
        + (" Broj parametara izražen je u milionima i zaokružen na tri decimale."
           if cfg.show_params else "")
        + (f" Kolone $\\Delta$ (sivo) prikazuju razliku u odnosu na {ref_label}."
           if ref else "")
        + " Strelice označavaju poželjan smer promene metrike.")
    label = cfg.label or f"tab:results_{cfg.scale}_{cfg.dataset}"
    prec = {m: prec_for(cfg.decimals, m) for m in cfg.metrics}

    # Najbolja vrednost po metrici (samo medju modelima, ne baznim).
    best = {}
    for m in cfg.metrics:
        vals = [rec[m] for _, rec in models]
        best[m] = max(vals) if higher_is_better(m) else min(vals)

    def cell(rec, m, bold=True):
        s = zf(rec[m], f".{prec[m]}f")
        if bold and cfg.bold and rec[m] == best[m]:
            s = r"\textbf{" + s + "}"
        return s

    def delta_cell(rec, m):
        # Zaokruzi pa odluci o predznaku, da ne izadje "-0,00"; znak u math modu,
        # broj van njega (zarez u math modu dodaje razmak).
        d = round(rec[m] - ref[m], prec[m])
        s = zf(abs(d), f".{prec[m]}f")
        if d > 0:
            s = "$+$" + s
        elif d < 0:
            s = "$-$" + s
        return r"\dlt{" + s + "}"

    def metric_cells(rec, bold=True):
        cells = []
        for m in cfg.metrics:
            cells.append(cell(rec, m, bold))
            if ref:
                cells.append(delta_cell(rec, m))
        return cells

    # Zaglavlje.
    headers = [r"\shortstack{Parametri\\($10^6$)}"] if cfg.show_params else []
    for i, m in enumerate(cfg.metrics):
        _, arrow, unit = METRICS[m]
        unit_line = r"\\(" + unit + ")" if cfg.header_units else ""
        gap = cfg.metric_gap + " " if i and cfg.metric_gap else ""
        if unit:
            headers.append(gap + r"\shortstack{" + f"{m} {arrow}" + unit_line + "}")
        else:
            headers.append(gap + f"{m} {arrow}")
        if ref:
            if unit:
                headers.append(r"\dlt{\shortstack{$\Delta$" + m + unit_line + "}}")
            else:
                headers.append(r"\dlt{$\Delta$" + m + "}")

    body = []
    for disp, rec in baselines:
        body.append(tex_row([disp] + (["---"] if cfg.show_params else []) + metric_cells(rec, bold=False)))
    if baselines:
        body.append(r"\midrule")
    for name, rec in models:
        cells = [display_name(name, cfg.show_config)]
        if cfg.show_params:
            cells.append(zf(rec['params'] / 1e6, f".{cfg.param_decimals}f") if rec.get("params") else "---")
        body.append(tex_row(cells + metric_cells(rec)))

    return tex_table("@{}l" + "r" * len(headers) + "@{}", [tex_row(["Model"] + headers)], body,
                     caption, label, group=True, tabcolsep="3.5pt" if ref else "4pt",
                     arraystretch="1.15",
                     preamble=[r"\newcommand*{\dlt}[1]{\textcolor{gray}{#1}}"] if ref else ())


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
