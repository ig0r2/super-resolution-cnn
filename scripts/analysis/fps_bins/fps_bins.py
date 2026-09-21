"""
fps_bins.py - najbolji modeli/arhitekture po opsezima brzine (FPS 480p).

Isto kao param_bins, ali umesto broja parametara modele deli po vrednosti FPS-a
(podrazumevano kolona "FPS 480p" iz istog results CSV-a) i ispisuje dve LaTeX
tabele:

  1) najbolji pojedinacni model po svakoj metrici (PSNR/SSIM/LPIPS) u svakom FPS
     opsegu (uz broj modela n, opciono);
  2) arhitektura sa najboljim prosekom metrike u svakom FPS opsegu.

Podrazumevano se izbacuju GAN/ESRGAN/jpeg i FastEDSR modeli. Ime modela je npr.
"EDSR_0_32" (bez SR_ prefiksa i scale tokena).

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Sve konfiguracije se obradjuju u jednom pokretanju; .tex se cuva u
results/analysis/fps_bins/.
"""

import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path import get_results_path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Metrika -> (broj decimala, da li je vece bolje).
METRIC_INFO = {
    "PSNR": (4, True),
    "SSIM": (4, True),
    "LPIPS": (4, False),
}
ARROW = {"PSNR": r"$\uparrow$", "SSIM": r"$\uparrow$", "LPIPS": r"$\downarrow$"}

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "Set14",  # koristi se za ime .tex fajla
    "scale": "2x",  # uvecanje (2x/3x/4x) — ulazi u ime CSV-a
    "dataset": "Set14",  # skup — ulazi u ime CSV-a
    "results": None,  # None -> results_{scale}_{dataset}_half.csv
    "bin_col": "FPS 480p",  # kolona po kojoj se binuje (brzina)
    "metrics": ["PSNR", "SSIM", "LPIPS"],  # metrike (kolone), redom
    "decimals": 3,  # broj decimala: None -> podrazumevano (4); int za sve;
    # ili dict npr. {"PSNR": 2, "SSIM": 4, "LPIPS": 3}
    "models": [],  # eksplicitna lista model_name; [] -> svi (posle exclude/archs)
    "archs": None,  # zadrzi samo ove arhitekture; None -> sve
    "show_n": False,  # True -> prikazi broj modela n; False -> bez n
    "exclude": ["GAN", "ESRGAN", "jpeg", "FastEDSR"],  # niske koje se izbacuju
    "bins": [  # (labela, donja granica FPS, gornja granica ili None)
        ("<30", 0, 30),
        ("30–45", 30, 45),
        ("45–60", 45, 60),
        ("60–90", 60, 90),
        ("90–120", 90, 120),
        ("120–250", 120, 250),
        ("250–500", 250, 500),
        (">500", 500, None),
    ],
    "out": None,  # None -> results/analysis/fps_bins/fps_bins_<name>.tex
}

##############################################


CONFIGS = [
    {
        "name": "Set14_2x_all",
        "scale": "2x",
        "dataset": "Set14",
        "metrics": ["SSIM", "LPIPS"],
        "exclude": ["GAN", "ESRGAN", "jpeg"],
    },
    {
        "name": "Set14_2x_standard",
        "scale": "2x",
        "metrics": ["SSIM", "LPIPS"],
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
    """SR_EDSR_2x_3_32 -> 'EDSR 3/32' (bez SR_ prefiksa, scale tokena i _r)."""
    parts = name.split("_")
    if parts and parts[0] == "SR":
        parts = parts[1:]
    parts = [p for p in parts if not re.fullmatch(r"\d+x", p) and p != "r"]
    if not parts:
        return name
    arch, rest = parts[0], parts[1:]
    return f"{arch} {'/'.join(rest)}" if rest else arch


def load_rows(path, metrics, bin_col, exclude, models, archs):
    """Vrati listu {name, arch, fps, <metrike>} za SR_ modele sa svim vrednostima."""
    models = set(models) if models else None
    archs = set(archs) if archs else None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = metrics + [bin_col]
        missing = [c for c in need if c not in reader.fieldnames]
        if missing:
            sys.exit(f"Kolone {', '.join(missing)} ne postoje u {path.name}.")
        for r in reader:
            name = r.get("model_name", "")
            if not name.startswith("SR_"):
                continue
            if models is not None:
                if name not in models:
                    continue
            else:
                if any(sub in name for sub in exclude):
                    continue
                if archs and arch_of(name) not in archs:
                    continue
            fps = to_float(r.get(bin_col))
            if not fps:
                continue
            rec = {"name": name, "arch": arch_of(name), "fps": fps}
            ok = True
            for m in metrics:
                v = to_float(r.get(m))
                if v is None:
                    ok = False
                    break
                rec[m] = v
            if ok:
                rows.append(rec)
    return rows


def in_bin(v, lo, hi):
    return v >= lo and (hi is None or v < hi)


def prec_for(cfg, metric):
    """Broj decimala za metriku prema cfg.decimals (None/int/dict)."""
    d = getattr(cfg, "decimals", None)
    if d is None:
        return METRIC_INFO[metric][0]
    if isinstance(d, dict):
        return d.get(metric, METRIC_INFO[metric][0])
    return d


def main():
    if not CONFIGS:
        sys.exit("CONFIGS je prazna - dodaj bar jednu konfiguraciju.")
    for i, cfg_dict in enumerate(CONFIGS):
        cfg = SimpleNamespace(**{**CONFIG_DEFAULTS, **cfg_dict})
        if i:
            print("\n" + "=" * 96 + "\n")
        print(f"### Konfiguracija: {cfg.name}\n")
        run_config(cfg)


def run_config(cfg):
    for m in cfg.metrics:
        if m not in METRIC_INFO:
            sys.exit(f"Nepoznata metrika '{m}'. Podrzane: {', '.join(METRIC_INFO)}")

    results_name = cfg.results or f"results_{cfg.scale}_{cfg.dataset}_half.csv"
    results_path = resolve_results(results_name)
    if not results_path.exists():
        sys.exit(f"CSV ne postoji: {results_path}")

    rows = load_rows(results_path, cfg.metrics, cfg.bin_col, cfg.exclude,
                     cfg.models, cfg.archs)
    if not rows:
        sys.exit(f"Nema modela u {results_path.name} sa vrednoscu '{cfg.bin_col}' "
                 f"posle filtera.")
    if cfg.models:
        found = {r["name"] for r in rows}
        for name in cfg.models:
            if name not in found:
                print(f"  [upozorenje] model '{name}' nije nadjen (ili nema {cfg.bin_col}) — preskacem")

    text = build_report(cfg, results_path, rows)
    print(text)

    out_path = Path(cfg.out) if cfg.out else get_results_path(
        f"analysis/fps_bins/fps_bins_{cfg.name}.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nLaTeX tabele: {out_path}")


def best_model(members, metric):
    _, higher = METRIC_INFO[metric]
    if not members:
        return None
    pick = (max if higher else min)(members, key=lambda r: r[metric])
    return display_name(pick["name"]), pick[metric]


def best_arch_avg(members, metric):
    _, higher = METRIC_INFO[metric]
    if not members:
        return None
    by_arch = {}
    for r in members:
        by_arch.setdefault(r["arch"], []).append(r[metric])
    stats = [(a, sum(v) / len(v), len(v)) for a, v in by_arch.items()]
    return (max if higher else min)(stats, key=lambda t: t[1])


def tex_label(s: str) -> str:
    """Labela FPS opsega -> LaTeX-safe (< > en-dash)."""
    return s.replace("<", r"$<$").replace(">", r"$>$").replace("–", "--")


def tex_name(s: str) -> str:
    return s.replace("_", r"\_")


def build_report(cfg, results_path, rows):
    tag = "" if cfg.models else (" bez FastEDSR" if "FastEDSR" in cfg.exclude else "")
    show_n = getattr(cfg, "show_n", True)
    prec = {m: prec_for(cfg, m) for m in cfg.metrics}
    scale_num = cfg.scale.rstrip("x")
    nmet = len(cfg.metrics)
    blocks = []

    # --- Tabela 1: najbolji model po metrici po FPS opsegu ---
    cols = "@{}l" + ("r" if show_n else "") + "l" * nmet + "@{}"
    header = [f"Opseg {cfg.bin_col}"] + (["$n$"] if show_n else [])
    header += [f"Najbolji {m} {ARROW[m]}" for m in cfg.metrics]
    t1 = [r"\begin{table}[H]", r"    \centering",
          f"    \\caption{{Najbolji model po metrici i opsegu brzine ({cfg.bin_col}) "
          f"({cfg.dataset}, $\\times{scale_num}${tag}).}}",
          f"    \\label{{tab:fps_bins_best_model_{cfg.scale}_{cfg.dataset}}}",
          r"    \small", r"    \setlength{\tabcolsep}{4pt}",
          f"    \\begin{{tabular}}{{{cols}}}", r"        \toprule",
          "        " + " & ".join(header) + r" \\", r"        \midrule"]
    for label, lo, hi in cfg.bins:
        members = [r for r in rows if in_bin(r["fps"], lo, hi)]
        cells = [tex_label(label)] + ([str(len(members))] if show_n else [])
        for m in cfg.metrics:
            b = best_model(members, m)
            cells.append(f"{tex_name(b[0])} ({zf(b[1], f'.{prec[m]}f')})" if b else "---")
        t1.append("        " + " & ".join(cells) + r" \\")
    t1 += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    blocks.append("\n".join(t1))

    # --- Tabela 2: najbolja arhitektura po proseku po FPS opsegu ---
    cols = "@{}l" + "l" * nmet + "@{}"
    header = [f"Opseg {cfg.bin_col}"] + [f"Najbolji prosek {m} {ARROW[m]}" for m in cfg.metrics]
    t2 = [r"\begin{table}[H]", r"    \centering",
          f"    \\caption{{Arhitektura sa najboljim prosekom metrike po opsegu brzine "
          f"({cfg.bin_col}) ({cfg.dataset}, $\\times{scale_num}${tag}).}}",
          f"    \\label{{tab:fps_bins_best_arch_{cfg.scale}_{cfg.dataset}}}",
          r"    \small", r"    \setlength{\tabcolsep}{4pt}",
          f"    \\begin{{tabular}}{{{cols}}}", r"        \toprule",
          "        " + " & ".join(header) + r" \\", r"        \midrule"]
    for label, lo, hi in cfg.bins:
        members = [r for r in rows if in_bin(r["fps"], lo, hi)]
        cells = [tex_label(label)]
        for m in cfg.metrics:
            b = best_arch_avg(members, m)
            if b:
                cells.append(f"{b[0]} ({zf(b[1], f'.{prec[m]}f')}, $n={b[2]}$)" if show_n
                             else f"{b[0]} ({zf(b[1], f'.{prec[m]}f')})")
            else:
                cells.append("---")
        t2.append("        " + " & ".join(cells) + r" \\")
    t2 += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    blocks.append("\n".join(t2))

    header_comment = f"% Izvor: {results_path.name}   modela: {len(rows)}   bin: {cfg.bin_col}"
    return header_comment + "\n\n" + "\n\n".join(blocks)


if __name__ == "__main__":
    main()
