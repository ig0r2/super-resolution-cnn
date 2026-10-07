"""
bins.py - najbolji modeli/arhitekture po opsezima broja parametara ili brzine.

Za zadati scale i dataset (config), modele iz results CSV-a deli u opsege (bins)
po vrednosti jedne kolone i ispisuje dve LaTeX tabele:

  1) najbolji pojedinacni model po svakoj metrici (PSNR/SSIM/LPIPS) u svakom
     opsegu (uz broj modela n, opciono);
  2) arhitektura sa najboljim prosekom metrike u svakom opsegu.

Vrsta opsega ("kind") odredjuje kolonu, podrazumevane opsege i izlaz:
    "params" - po broju parametara -> results/analysis/param_bins/param_bins_<name>.tex
    "fps"    - po brzini (FPS 480p) -> results/analysis/fps_bins/fps_bins_<name>.tex

Podrazumevano se izbacuju GAN/ESRGAN/jpeg i FastEDSR modeli (za "fps" i ABPN).
Ime modela je npr. "EDSR 0/32" (bez SR_ prefiksa i scale tokena).

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Sve konfiguracije se obradjuju u jednom pokretanju.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.config import run_configs
from utils.analysis.data import read_rows, require_results, results_csv, to_float
from utils.analysis.fmt import tex_model, tex_row, tex_table, zf
from utils.analysis.metrics import arrow, best_by, check_metrics, prec_for
from utils.analysis.names import arch_of, display_name, is_multiscale
from utils.path import get_results_path

# Podrazumevane vrednosti zajednicke za obe vrste. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima iz KIND_DEFAULTS[kind] pa odavde.
CONFIG_DEFAULTS = {
    "name": "default",  # koristi se za ime .tex fajla
    "kind": "params",  # "params" (opsezi broja parametara) ili "fps" (opsezi brzine)
    "scale": "2x",  # uvecanje (2x/3x/4x) — ulazi u ime CSV-a
    "dataset": "DIV2K",  # skup — ulazi u ime CSV-a
    "results": None,  # None -> results_{scale}_{dataset}_half.csv
    "metrics": ["PSNR", "SSIM", "LPIPS"],  # metrike (kolone), redom
    "models": [],  # eksplicitna lista model_name; [] -> svi (posle exclude/archs)
    "archs": None,  # zadrzi samo ove arhitekture (npr. ["RFDN"]); None -> sve
    "show_n": False,  # True -> prikazi broj modela n; False -> bez n
    "include_multiscale": True,  # True -> ulaze i modeli bez \d+x tokena u imenu
    # (multiscale), pored onih sa cfg.scale
    "out": None,  # None -> results/analysis/<prefiks>/<prefiks>_<name>.tex (KIND_PREFIX)
}

# Vrsta -> prefiks izlaznog foldera/fajla i LaTeX labela.
KIND_PREFIX = {"params": "param_bins", "fps": "fps_bins"}

KIND_DEFAULTS = {
    "params": {
        "bin_col": "params",  # kolona po kojoj se binuje
        "decimals": {"PSNR": 2, "SSIM": 3, "LPIPS": 3},  # broj decimala: None -> podrazumevano (4);
        # int za sve; ili dict npr. {"PSNR": 2, "SSIM": 4, "LPIPS": 3}
        "exclude": ["GAN", "ESRGAN", "jpeg"],  # niske koje se izbacuju
        "bins": [  # (labela, donja granica, gornja granica ili None)
            ("<10K", 0, 10_000),
            ("10K-50K", 10_000, 50_000),
            ("50K–200K", 50_000, 200_000),
            ("200K–500K", 200_000, 500_000),
            ("500K–1M", 500_000, 1_000_000),
            ("1M–3M", 1_000_000, 3_000_000),
            (">3M", 3_000_000, None),
        ],
    },
    "fps": {
        "bin_col": "FPS 480p",
        "decimals": 3,
        "exclude": ["GAN", "ESRGAN", "jpeg"],
        "bins": [
            ("<30", 0, 30),
            ("30–45", 30, 45),
            ("45–60", 45, 60),
            ("60–90", 60, 90),
            ("90–120", 90, 120),
            ("120–250", 120, 250),
            ("250–500", 250, 500),
            (">500", 500, None),
        ],
    },
}

##############################################


CONFIGS = [
    # --- opsezi broja parametara ---
    {
        "name": "DIV2K_2x",
        "kind": "params",
    },
    # --- opsezi brzine (FPS 480p) ---
    {
        "name": "DIV2K_2x",
        "kind": "fps",
        "metrics": ["SSIM", "LPIPS"],
    },
]


##############################################


def defaults_for(cfg_dict):
    kind = cfg_dict.get("kind", CONFIG_DEFAULTS["kind"])
    if kind not in KIND_DEFAULTS:
        sys.exit(f"Nepoznat kind '{kind}'. Podrzano: {', '.join(KIND_DEFAULTS)}")
    return {**CONFIG_DEFAULTS, **KIND_DEFAULTS[kind]}


def load_rows(path, cfg):
    """Vrati listu {name, arch, v, <metrike>} za SR_ modele sa svim vrednostima (v = bin_col).

    Ako je `models` neprazna lista -> zadrzavaju se tacno ti modeli (exclude/archs
    se ignorisu). Inace se primenjuju exclude, (opciono) archs i scale filter.
    """
    models = set(cfg.models) if cfg.models else None
    archs = set(cfg.archs) if cfg.archs else None
    rows = []
    for r in read_rows(path, cfg.metrics + [cfg.bin_col]):
        name = r.get("model_name", "")
        if not name.startswith("SR_"):
            continue
        if models is not None:
            if name not in models:
                continue
        else:
            if any(sub in name for sub in cfg.exclude):
                continue
            if archs and arch_of(name) not in archs:
                continue
            if cfg.scale not in name.split("_"):
                if not (cfg.include_multiscale and is_multiscale(name)):
                    continue
        v = to_float(r.get(cfg.bin_col))
        if not v:
            continue
        rec = {"name": name, "arch": arch_of(name), "v": v}
        rec.update((m, to_float(r.get(m))) for m in cfg.metrics)
        if all(rec[m] is not None for m in cfg.metrics):
            rows.append(rec)
    return rows


def in_bin(v, lo, hi):
    return v >= lo and (hi is None or v < hi)


def best_arch_avg(members, metric):
    """Arhitektura sa najboljim prosekom metrike -> (arch, avg, n) ili None."""
    by_arch = {}
    for r in members:
        by_arch.setdefault(r["arch"], []).append(r[metric])
    stats = [(a, sum(v) / len(v), len(v)) for a, v in by_arch.items()]
    return best_by(stats, metric, key=lambda t: t[1])


def tex_label(s: str) -> str:
    """Labela opsega -> LaTeX-safe (< > en-dash, razmak ispred K/M)."""
    return (s.replace("<", r"$<$").replace(">", r"$>$")
            .replace("–", "--").replace("K", r"\,K").replace("M", r"\,M"))


def run_config(cfg):
    check_metrics(cfg.metrics)
    results_path = require_results(cfg.results or results_csv(cfg.scale, cfg.dataset))

    rows = load_rows(results_path, cfg)
    if not rows:
        sys.exit(f"Nema modela u {results_path.name} sa vrednoscu '{cfg.bin_col}' posle filtera.")
    if cfg.models:
        found = {r["name"] for r in rows}
        for name in cfg.models:
            if name not in found:
                print(f"  [upozorenje] model '{name}' nije nadjen (ili nema {cfg.bin_col}) — preskacem")

    text = build_report(cfg, results_path, rows)
    print(text)

    out_path = Path(cfg.out) if cfg.out else get_results_path(
        f"analysis/{KIND_PREFIX[cfg.kind]}/{KIND_PREFIX[cfg.kind]}_{cfg.name}.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nLaTeX tabele: {out_path}")


def build_report(cfg, results_path, rows):
    tag = "" if cfg.models else (" bez FastEDSR" if "FastEDSR" in cfg.exclude else "")
    prec = {m: prec_for(cfg.decimals, m) for m in cfg.metrics}
    where = f"({cfg.dataset}, $\\times{cfg.scale.rstrip('x')}${tag})"
    if cfg.kind == "params":
        range_desc, range_head, comment = "parametara", "Opseg parametara", ""
    else:
        range_desc, range_head, comment = f"brzine ({cfg.bin_col})", f"Opseg {cfg.bin_col}", f"   bin: {cfg.bin_col}"
    label_prefix = KIND_PREFIX[cfg.kind]
    nmet = len(cfg.metrics)
    bins = [(label, [r for r in rows if in_bin(r["v"], lo, hi)]) for label, lo, hi in cfg.bins]

    # --- Tabela 1: najbolji model po metrici po opsegu ---
    header = [range_head] + (["$n$"] if cfg.show_n else [])
    header += [f"Najbolji {m} {arrow(m)}" for m in cfg.metrics]
    body = []
    for label, members in bins:
        cells = [tex_label(label)] + ([str(len(members))] if cfg.show_n else [])
        for m in cfg.metrics:
            b = best_by(members, m)
            if b:
                arch, _, conf = display_name(b["name"]).partition(" ")
                cells.append(f"{tex_model(arch, conf)} ({zf(b[m], f'.{prec[m]}f')})")
            else:
                cells.append("---")
        body.append(tex_row(cells))
    t1 = tex_table("@{}l" + ("r" if cfg.show_n else "") + "l" * nmet + "@{}", [tex_row(header)], body,
                   f"Najbolji model po metrici i opsegu {range_desc} {where}.",
                   f"tab:{label_prefix}_best_model_{cfg.scale}_{cfg.dataset}")

    # --- Tabela 2: najbolja arhitektura po proseku po opsegu ---
    header = [range_head] + [f"Najbolji prosek {m} {arrow(m)}" for m in cfg.metrics]
    body = []
    for label, members in bins:
        cells = [tex_label(label)]
        for m in cfg.metrics:
            b = best_arch_avg(members, m)
            if b:
                n = f", $n={b[2]}$" if cfg.show_n else ""
                cells.append(f"{b[0]} ({zf(b[1], f'.{prec[m]}f')}{n})")
            else:
                cells.append("---")
        body.append(tex_row(cells))
    t2 = tex_table("@{}l" + "l" * nmet + "@{}", [tex_row(header)], body,
                   f"Arhitektura sa najboljim prosekom metrike po opsegu {range_desc} {where}.",
                   f"tab:{label_prefix}_best_arch_{cfg.scale}_{cfg.dataset}")

    header_comment = f"% Izvor: {results_path.name}   modela: {len(rows)}{comment}"
    return header_comment + "\n\n" + t1 + "\n\n" + t2


if __name__ == "__main__":
    run_configs(CONFIGS, defaults_for, run_config)
