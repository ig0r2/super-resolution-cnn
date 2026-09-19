"""
jpeg_metrics.py - poredjenje baznog modela i njegovog jpeg parnjaka.

Za svaki model koji ima jpeg parnjaka (npr. SR_FastEDSR_4_64 <-> SR_FastEDSR_
jpeg_4_64) u zadatom CSV-u, ispisuje tabelu sa metrikama baznog i jpeg modela i
njihovom razlikom (Δ = jpeg - bazni). Uz obicnu tabelu, izbacuje i gotovu LaTeX
tabelu koja se moze direktno prekopirati u tex dokument (bolduje bolju vrednost
po metrici).
"""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.logger import Logger
from utils.path import get_results_path

# Metrike: (broj decimala, da li je vece bolje, strelica za LaTeX header).
METRIC_INFO = {
    "LPIPS": (4, False, r"$\downarrow$"),
    "SSIM": (4, True, r"$\uparrow$"),
    "PSNR": (2, True, r"$\uparrow$"),
}

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "Set14_jpeg",  # koristi se za imena izlaznih fajlova
    "results": "results_2x_Set14_jpeg_half.csv",  # CSV sa evaluacijom
    "pairing": "base_vs_jpeg",  # "base_vs_jpeg" (bazni vs jpeg) ili
                                # "jpeg_vs_s" (jpeg vs jpeg_s sufiks)
    "metrics": ["PSNR", "SSIM", "LPIPS"],  # metrike za poredjenje (redom)
    "out": None,   # None -> results/analysis/jpeg/jpeg_metrics_<name>.txt
    "tex_out": None,  # None -> results/analysis/jpeg/jpeg_metrics_<name>.tex
    "no_log": False,  # True -> samo ispis na ekran, bez fajlova
}

##############################################


CONFIGS = [
    {
        "name": "Set14_jpeg",
        "pairing": "base_vs_jpeg",
    },
    {
        "name": "Set14_jpeg_vs_s",
        "pairing": "jpeg_vs_s",
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


def jpeg_partner(name: str) -> str | None:
    """SR_<arh>_<konfig> -> SR_<arh>_jpeg_<konfig>; None ako ime nije SR_ oblik."""
    parts = name.split("_")
    if len(parts) < 3 or parts[0] != "SR" or "jpeg" in parts:
        return None
    return "_".join(parts[:2] + ["jpeg"] + parts[2:])


def s_partner(name: str) -> str | None:
    """jpeg model -> isti + "_s"; None ako nije jpeg ili vec ima _s sufiks."""
    if "jpeg" not in name.split("_") or name.endswith("_s"):
        return None
    return name + "_s"


# Rezim uparivanja -> (funkcija za parnjaka, oznaka leve kolone, oznaka desne).
PAIRINGS = {
    "base_vs_jpeg": (jpeg_partner, "bazni", "jpeg"),
    "jpeg_vs_s": (s_partner, "jpeg", "jpeg_s"),
}


def load_rows(path, metrics):
    """Vrati {model_name: {metric: value}} za redove koji imaju sve metrike."""
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
            rec = {}
            for m in metrics:
                v = to_float(r.get(m))
                if v is not None:
                    rec[m] = v
            if len(rec) == len(metrics):
                out[name] = rec
    return out


def short_name(name: str) -> str:
    """Ime za prikaz: bez SR_ prefiksa i bez 'jpeg' tokena."""
    parts = [p for p in name.split("_") if p != "jpeg"]
    if parts and parts[0] == "SR":
        parts = parts[1:]
    return "_".join(parts)


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
    if cfg.pairing not in PAIRINGS:
        sys.exit(f"Nepoznat pairing '{cfg.pairing}'. Podrzano: {', '.join(PAIRINGS)}")
    partner_fn, left_label, right_label = PAIRINGS[cfg.pairing]

    results_path = resolve_results(cfg.results)
    if not results_path.exists():
        sys.exit(f"CSV ne postoji: {results_path}")

    rows = load_rows(results_path, cfg.metrics)

    # Parovi (levi, desni) po imenu, sortirani po imenu levog modela.
    pairs = []
    for name in sorted(rows):
        partner = partner_fn(name)
        if partner and partner in rows:
            pairs.append((name, partner))
    if not pairs:
        sys.exit(f"Nema modela sa parnjakom za pairing '{cfg.pairing}' u CSV-u.")

    if cfg.no_log:
        report(cfg, results_path, rows, pairs, left_label, right_label)
    else:
        txt_path = Path(cfg.out) if cfg.out else get_results_path(
            f"analysis/jpeg/jpeg_metrics_{cfg.name}.txt")
        with Logger(txt_path):
            report(cfg, results_path, rows, pairs, left_label, right_label)
        # LaTeX tabela u zaseban .tex (a ista je i u .txt iznad).
        tex_path = Path(cfg.tex_out) if cfg.tex_out else get_results_path(
            f"analysis/jpeg/jpeg_metrics_{cfg.name}.tex")
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(tex_table(cfg, rows, pairs, left_label, right_label),
                            encoding="utf-8")
        print(f"LaTeX tabela: {tex_path}")


def report(cfg, results_path, rows, pairs, left_label, right_label):
    print(f"Izvor: {results_path.name}")
    print(f"Parova ({left_label} vs {right_label}): {len(pairs)}   "
          f"(d = {right_label} - {left_label})")
    print()

    # Zaglavlje: za svaku metriku tri kolone (levi, desni, d = razlika).
    head = f"{'Model':<26}"
    for m in cfg.metrics:
        head += f" {m+' '+left_label:>12} {m+' '+right_label:>12} {'d'+m:>11}"
    print(head)
    print("-" * len(head))

    for base, jp in pairs:
        line = f"{short_name(base):<26}"
        for m in cfg.metrics:
            prec, _, _ = METRIC_INFO[m]
            b = rows[base][m]
            j = rows[jp][m]
            line += f" {b:>12.{prec}f} {j:>12.{prec}f} {j - b:>+11.{prec}f}"
        print(line)

    # Prosecna razlika po metrici.
    print("-" * len(head))
    avg = f"{'PROSEK d':<26}"
    for m in cfg.metrics:
        prec, _, _ = METRIC_INFO[m]
        deltas = [rows[j][m] - rows[b][m] for b, j in pairs]
        blank = " " * 12
        avg += f" {blank} {blank} {sum(deltas) / len(deltas):>+11.{prec}f}"
    print(avg)
    print()

    print("LaTeX tabela (za kopiranje):")
    print(tex_table(cfg, rows, pairs, left_label, right_label))


def tex_table(cfg, rows, pairs, left_label, right_label):
    """Vrati LaTeX (booktabs) tabelu; bolduje bolju vrednost po metrici."""
    ncol = len(cfg.metrics)
    lines = []
    lines.append(r"\begin{tabular}{l" + "cc" * ncol + "}")
    lines.append(r"\toprule")

    # Grupisano zaglavlje: po dve kolone (bazni/jpeg) za svaku metriku.
    top = ["Model"]
    for m in cfg.metrics:
        _, _, arrow = METRIC_INFO[m]
        top.append(r"\multicolumn{2}{c}{" + f"{m} {arrow}" + "}")
    lines.append(" & ".join(top) + r" \\")
    cmids = "".join(r"\cmidrule(lr){" + f"{2 + 2 * k}-{3 + 2 * k}" + "}"
                    for k in range(ncol))
    lines.append(cmids)
    sub = [""] + [f"{left_label} & {right_label}".replace("_", r"\_")] * ncol
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    def fmt(v, prec, better):
        s = f"{v:.{prec}f}"
        return r"\textbf{" + s + "}" if better else s

    for base, jp in pairs:
        cells = [short_name(base).replace("_", r"\_")]
        for m in cfg.metrics:
            prec, higher_better, _ = METRIC_INFO[m]
            b = rows[base][m]
            j = rows[jp][m]
            j_better = (j > b) if higher_better else (j < b)
            cells.append(fmt(b, prec, not j_better and b != j))
            cells.append(fmt(j, prec, j_better and b != j))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
