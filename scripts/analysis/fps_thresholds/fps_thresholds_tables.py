"""
fps_thresholds_tables.py - najbolji modeli/arhitekture uz zadatu minimalnu brzinu.

Za razliku od bins (opsezi brzine), ovde se za svaki prag T posmatraju SVI
modeli sa FPS >= T, pa se bira najbolji. To direktno odgovara na pitanje "imam
budzet od T FPS, koji je najbolji model koji ga dostize" - model iz brzeg opsega
nije iskljucen samo zato sto je brzi nego sto treba.

Za zadati scale i dataset (config) ispisuje LaTeX tabele:

  1) "Pregled" - za svaki prag najbolji model po svakoj metrici.
  2) "Po metrici" - redovi su serije (arhitekture), kolone pragovi, a celija je
     najbolja vrednost te arhitekture uz FPS >= prag (najbolja u koloni bold).

Serija je ime arhitekture iz model_name (SR_EDSR_2x_16_64 -> "EDSR"). Viseskalni
modeli (bez Nx tokena u imenu, npr. SR_FastEDSR_4_64) dobijaju sufiks "-M"
("FastEDSR-M") i ulaze samo uz include_multiscale=True.

Podesavanje ide iskljucivo preko CONFIGS liste ispod - bez argumenata komandne
linije. Sve konfiguracije se obradjuju u jednom pokretanju; .tex se cuva u
results/analysis/fps_thresholds/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.config import run_configs
from utils.analysis.data import read_rows, require_results, results_csv, to_float
from utils.analysis.fmt import tex_model, tex_name, tex_row, tex_table, zf
from utils.analysis.metrics import arrow, best_by, check_metrics, prec_for
from utils.analysis.names import series_name
from utils.path import get_results_path

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "default",  # koristi se za ime .tex fajla i labele tabela
    "scale": "2x",  # uvecanje (2x/3x/4x) — ulazi u ime CSV-a
    "dataset": "DIV2K",  # skup — ulazi u ime CSV-a
    "results": None,  # None -> results_{scale}_{dataset}_half.csv
    "fps_col": "FPS 480p",  # kolona sa brzinom
    "metrics": ["SSIM", "LPIPS"],  # metrike (kolone), redom
    "decimals": {"PSNR": 2, "SSIM": 4, "LPIPS": 3},  # broj decimala: None -> podrazumevano (4); int za sve;
    # ili dict npr. {"PSNR": 2, "SSIM": 4, "LPIPS": 3}
    "thresholds": [20, 30, 60, 80, 120, 250, 500],  # pragovi FPS-a
    "series": None,  # serije, tim redom (npr. ["EDSR", "FastEDSR-M"]); None -> sve
    "exclude": ["GAN", "ESRGAN", "jpeg"],  # niske koje se izbacuju
    "include_multiscale": False,  # True -> ulaze i viseskalni modeli (serije "-M")
    "show_config": False,  # u tabeli po metrici prikazi i konfiguraciju (B/n_f)
    "best_config": False,  # u tabeli po metrici: konfiguracija kao indeks samo uz podebljanu (najbolju) celiju
    "summary": True,  # tabela 1 (pregled)
    "per_metric": True,  # tabele 2 (po metrici)
    "out": None,  # None -> results/analysis/fps_thresholds/thresholds_<name>.tex
}

##############################################

LITERATURE = ["VDSR", "EDSR", "IMDN", "RFDN"]

CONFIGS = [
    {
        # Namenski x2 modeli (za poglavlje o arhitekturama).
        "name": "single",
        "series": LITERATURE + ["FastEDSR", "ABPN"],
    },
]


##############################################


def load_rows(path, cfg):
    """Vrati listu {name, series, config, fps, <metrike>} za SR_ modele sa svim metrikama i FPS-om."""
    rows = []
    for r in read_rows(path, cfg.metrics + [cfg.fps_col]):
        name = r.get("model_name", "")
        if not name.startswith("SR_") or any(sub in name for sub in cfg.exclude):
            continue
        s, config = series_name(name)
        if s.endswith("-M") and not cfg.include_multiscale:
            continue
        if cfg.series is not None and s not in cfg.series:
            continue
        fps = to_float(r.get(cfg.fps_col))
        if not fps:
            continue
        rec = {"name": name, "series": s, "config": config, "fps": fps}
        rec.update((m, to_float(r.get(m))) for m in cfg.metrics)
        if all(rec[m] is not None for m in cfg.metrics):
            rows.append(rec)
    return rows


def run_config(cfg):
    check_metrics(cfg.metrics)
    results_path = require_results(cfg.results or results_csv(cfg.scale, cfg.dataset))

    rows = load_rows(results_path, cfg)
    if not rows:
        sys.exit(f"Nema modela u {results_path.name} posle filtera.")
    present = {r["series"] for r in rows}
    if cfg.series is not None:
        for s in cfg.series:
            if s not in present and not (s.endswith("-M") and not cfg.include_multiscale):
                print(f"  [upozorenje] serija '{s}' nema nijedan model sa FPS vrednoscu — preskacem")
        series = [s for s in cfg.series if s in present]
    else:
        series = sorted(present)

    text = build_report(cfg, results_path, rows, series)
    print(text)

    out_path = Path(cfg.out) if cfg.out else get_results_path(
        f"analysis/fps_thresholds/thresholds_{cfg.name}.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nLaTeX tabele: {out_path}")


def build_report(cfg, results_path, rows, series):
    prec = {m: prec_for(cfg.decimals, m) for m in cfg.metrics}
    where = f"{cfg.fps_col}; {cfg.dataset}, $\\times{cfg.scale.rstrip('x')}$"

    blocks = []

    # --- Tabela 1: najbolji model po metrici uz FPS >= prag ---
    if cfg.summary:
        header = ["FPS $\\geq$"] + [f"Najbolji {m} {arrow(m)}" for m in cfg.metrics]
        body = []
        for thr in cfg.thresholds:
            members = [r for r in rows if r["fps"] >= thr]
            cells = [zf(thr, "g")]
            for m in cfg.metrics:
                b = best_by(members, m)
                cells.append(f"{tex_model(b['series'], b['config'])} ({zf(b[m], f'.{prec[m]}f')})"
                             if b else "---")
            body.append(tex_row(cells))
        blocks.append(tex_table(
            "@{}l" + "l" * len(cfg.metrics) + "@{}", [tex_row(header)], body,
            f"Najbolji model po metrici uz zadatu minimalnu brzinu ({where}).",
            f"tab:fps_thr_summary_{cfg.name}"))

    # --- Tabele 2: po metrici, serije x pragovi ---
    if cfg.per_metric:
        header = ["FPS"] + [f"$\\geq${zf(thr, 'g')}" for thr in cfg.thresholds]
        for m in cfg.metrics:
            col_best = {thr: best_by([r for r in rows if r["fps"] >= thr], m) for thr in cfg.thresholds}
            body = []
            for s in series:
                cells = [tex_name(s)]
                for thr in cfg.thresholds:
                    b = best_by([r for r in rows if r["series"] == s and r["fps"] >= thr], m)
                    if not b:
                        cells.append("---")
                        continue
                    v = zf(b[m], f".{prec[m]}f")
                    if col_best[thr] is b:
                        v = f"\\textbf{{{v}}}"
                        if cfg.best_config and not cfg.show_config and b["config"]:
                            v += f"$_{{\\text{{{b['config']}}}}}$"
                    if cfg.show_config:
                        v += f"\\,{{\\scriptsize({b['config']})}}"
                    cells.append(v)
                body.append(tex_row(cells))
            caption = (f"Najbolji {m} {arrow(m)} koji svaka arhitektura postiže uz "
                       f"zadatu minimalnu brzinu ({where})"
                       + (", uz konfiguraciju $B/n_f$" if cfg.show_config else "")
                       + ". Najbolja vrednost u koloni je podebljana"
                       + (", uz konfiguraciju $B/n_f$ u indeksu" if cfg.best_config and not cfg.show_config else "")
                       + ".")
            blocks.append(tex_table(
                "@{}l" + "r" * len(cfg.thresholds) + "@{}", [tex_row(header)], body,
                caption, f"tab:fps_thr_{m}_{cfg.name}", size="footnotesize", tabcolsep="3pt"))

    header_comment = f"% Izvor: {results_path.name}   modela: {len(rows)}   brzina: {cfg.fps_col}"
    return header_comment + "\n\n" + "\n\n".join(blocks)


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
