"""
jpeg_table.py - poredjenje baznog modela i njegovog jpeg parnjaka.

Za svaki model koji ima jpeg parnjaka (npr. SR_FastEDSR_4_64 <-> SR_FastEDSR_
jpeg_4_64) u zadatom CSV-u, ispisuje tabelu sa metrikama baznog i jpeg modela i
njihovom razlikom (Δ = jpeg - bazni). Uz obicnu tabelu, izbacuje i gotovu LaTeX
tabelu koja se moze direktno prekopirati u tex dokument (bolduje bolju vrednost
po metrici).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.config import run_configs
from utils.analysis.data import read_rows, require_results, to_float
from utils.analysis.fmt import tex_name, tex_row, tex_table, zf
from utils.analysis.metrics import arrow, check_metrics, higher_is_better, prec_for
from utils.analysis.names import display_name
from utils.path import get_results_path

# Podrazumevan broj decimala (kad "decimals" ne zadaje drugacije); ostale metrike 4.
DEFAULT_DECIMALS = {"PSNR": 2}

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "DIV2K_jpeg",  # koristi se za imena izlaznih fajlova
    "results": "results_2x_DIV2K_jpeg_half.csv",  # CSV sa evaluacijom
    "pairing": "base_vs_jpeg",  # "base_vs_jpeg" (bazni vs jpeg) ili
    # "jpeg_vs_s" (jpeg vs jpeg_s sufiks)
    "metrics": ["PSNR", "SSIM", "LPIPS"],  # metrike za poredjenje (redom)
    "show_base": False,  # True (samo za "jpeg_vs_s") -> dodaj i kolonu baznog modela
    "decimals": 3,  # broj decimala: None -> podrazumevano (PSNR 2, ostalo 4);
    # int za sve; ili dict npr. {"PSNR": 2, "SSIM": 4, "LPIPS": 4}
    "caption": "Poređenje JPEG modela",  # \caption tabele
    "label": None,  # None -> tab:jpeg_<name>
    "tex_out": None,  # None -> results/analysis/jpeg/jpeg_table_<name>.tex
}

##############################################


CONFIGS = [
    {
        "name": "DIV2K_jpeg",
        "pairing": "base_vs_jpeg",
    },
    {
        "name": "DIV2K_jpeg_vs_s",
        "metrics": ["SSIM", "LPIPS"],
        "pairing": "jpeg_vs_s",
        "show_base": True
    },
    {
        "name": "DIV2K_jpeg45",
        "results": "results_2x_DIV2K_jpeg45_half.csv",
        "pairing": "base_vs_jpeg",
    },
    {
        "name": "DIV2K_jpeg80",
        "results": "results_2x_DIV2K_jpeg80_half.csv",
        "pairing": "base_vs_jpeg",
    },
    {
        "name": "DIV2K_jpeg80_vs_s",
        "results": "results_2x_DIV2K_jpeg80_half.csv",
        "pairing": "jpeg_vs_s",
        "show_base": True
    },
    # 4x
    {
        "name": "4x_DIV2K_jpeg",
        "results": "results_4x_DIV2K_jpeg_half.csv",
        "pairing": "base_vs_jpeg",
    },
    {
        "name": "4x_DIV2K_jpeg_vs_s",
        "results": "results_4x_DIV2K_jpeg_half.csv",
        "metrics": ["SSIM", "LPIPS"],
        "pairing": "jpeg_vs_s",
        "show_base": True
    },
    {
        "name": "4x_DIV2K_jpeg45",
        "results": "results_4x_DIV2K_jpeg45_half.csv",
        "pairing": "base_vs_jpeg",
    },
    {
        "name": "4x_DIV2K_jpeg80",
        "results": "results_4x_DIV2K_jpeg80_half.csv",
        "pairing": "base_vs_jpeg",
    },
    {
        "name": "4x_DIV2K_jpeg80_vs_s",
        "results": "results_4x_DIV2K_jpeg80_half.csv",
        "metrics": ["SSIM", "LPIPS"],
        "pairing": "jpeg_vs_s",
        "show_base": True
    },
]


##############################################


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
    """Vrati ({model_name: {metric: value}}, {model_name: params}) za pune redove."""
    out, params = {}, {}
    for r in read_rows(path, metrics):
        name = r.get("model_name", "")
        if not name:
            continue
        rec = {m: to_float(r.get(m)) for m in metrics}
        if all(v is not None for v in rec.values()):
            out[name] = rec
            params[name] = to_float(r.get("params"))
    return out, params


def base_name(name: str) -> str:
    """jpeg model -> bazni parnjak (bez 'jpeg' tokena), zadrzava SR_ prefiks."""
    return "_".join(p for p in name.split("_") if p != "jpeg")


def short_name(name: str) -> str:
    """Ime za prikaz: 'SR_FastEDSR_jpeg_2_64' -> 'FastEDSR 2/64' (bez jpeg/scale/_r)."""
    return display_name(base_name(name))


def disp_labels(left_label, right_label, show_base):
    """Labele kolona vrednosti (redom); opciono i 'bazni' na pocetku."""
    return (["bazni"] if show_base else []) + [left_label, right_label]


def disp_names(left_name, right_name, show_base):
    """Imena modela za kolone vrednosti (redom), uparena sa disp_labels."""
    return ([base_name(left_name)] if show_base else []) + [left_name, right_name]


def prec(cfg, metric):
    return prec_for(cfg.decimals, metric, DEFAULT_DECIMALS)


def run_config(cfg):
    check_metrics(cfg.metrics)
    if cfg.pairing not in PAIRINGS:
        sys.exit(f"Nepoznat pairing '{cfg.pairing}'. Podrzano: {', '.join(PAIRINGS)}")
    partner_fn, left_label, right_label = PAIRINGS[cfg.pairing]
    # Bazna kolona ima smisla samo kad su levi modeli jpeg (pairing jpeg_vs_s).
    show_base = cfg.show_base and cfg.pairing == "jpeg_vs_s"

    results_path = require_results(cfg.results)
    rows, params = load_rows(results_path, cfg.metrics)

    # Parovi (levi, desni), sortirani po broju parametara levog modela (rastuce).
    pairs = []
    for name in rows:
        partner = partner_fn(name)
        if partner and partner in rows:
            pairs.append((name, partner))
    if not pairs:
        sys.exit(f"Nema modela sa parnjakom za pairing '{cfg.pairing}' u CSV-u.")
    pairs.sort(key=lambda p: (params.get(p[0]) is None, params.get(p[0]) or 0.0, p[0]))

    report(cfg, results_path, rows, pairs, left_label, right_label, show_base)

    # LaTeX tabela u zaseban .tex (a ista je i u ispisu iznad).
    tex_path = Path(cfg.tex_out) if cfg.tex_out else get_results_path(
        f"analysis/jpeg/jpeg_table_{cfg.name}.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(build_table(cfg, rows, pairs, left_label, right_label, show_base),
                        encoding="utf-8")
    print(f"LaTeX tabela: {tex_path}")


def best_val(rows, names, m):
    """Najbolja vrednost metrike m medju prisutnim modelima iz `names`."""
    present = [rows[n][m] for n in names if n in rows]
    if not present:
        return None
    return (max if higher_is_better(m) else min)(present)


def avg_delta_map(rows, pairs, metrics, a_of, b_of):
    """Prosecna razlika (a - b) po metrici; a_of/b_of daju imena modela iz para."""
    vals = {m: [] for m in metrics}
    for left, right in pairs:
        a, b = a_of(left, right), b_of(left, right)
        if a in rows and b in rows:
            for m in metrics:
                vals[m].append(rows[a][m] - rows[b][m])
    return {m: (sum(vals[m]) / len(vals[m]) if vals[m] else None) for m in metrics}


def _delta_items_plain(cfg, avg):
    return ",  ".join(f"{m} {zf(avg[m], f'+.{prec(cfg, m)}f')}"
                      for m in cfg.metrics if avg[m] is not None)


def _delta_items_tex(cfg, avg):
    items = []
    for m in cfg.metrics:
        if avg[m] is None:
            continue
        sign = "-" if avg[m] < 0 else "+"
        items.append(f"{m} ${sign}${zf(abs(avg[m]), f'.{prec(cfg, m)}f')}")
    return ", ".join(items)


def _avg_deltas(cfg, rows, pairs, show_base):
    """[(oznaka umanjioca ili None za levi, prosecne razlike)] za konzolu i LaTeX."""
    out = [(None, avg_delta_map(rows, pairs, cfg.metrics, lambda l, r: r, lambda l, r: l))]
    if show_base:
        out.append(("bazni", avg_delta_map(rows, pairs, cfg.metrics, lambda l, r: r,
                                           lambda l, r: base_name(l))))
    return out


def avg_delta_line(cfg, rows, pairs, left_label, right_label, show_base):
    """Tekstualne recenice sa prosecnom razlikom po metrici (za konzolu)."""
    return "\n".join(f"Prosecna razlika ({right_label} - {sub or left_label}) po metrici: "
                     + _delta_items_plain(cfg, avg)
                     for sub, avg in _avg_deltas(cfg, rows, pairs, show_base))


def avg_delta_tex(cfg, rows, pairs, left_label, right_label, show_base):
    """LaTeX recenice sa prosecnom razlikom po metrici (ispod tabele)."""
    return "\n\n".join(r"\noindent Prosečna razlika (" + tex_name(right_label) + r" $-$ "
                       + tex_name(sub or left_label) + "): " + _delta_items_tex(cfg, avg) + "."
                       for sub, avg in _avg_deltas(cfg, rows, pairs, show_base))


def report(cfg, results_path, rows, pairs, left_label, right_label, show_base):
    labels = disp_labels(left_label, right_label, show_base)
    print(f"Izvor: {results_path.name}")
    print(f"Parova ({left_label} vs {right_label}): {len(pairs)}   "
          f"(d = {right_label} - {left_label})"
          + ("   [+ bazni]" if show_base else ""))
    print()

    # Zaglavlje: za svaku metriku po kolona vrednosti (labels) + d = razlika.
    head = f"{'Model':<26}"
    for m in cfg.metrics:
        for lab in labels:
            head += f" {m + ' ' + lab:>12}"
        head += f" {'d' + m:>11}"
    print(head)
    print("-" * len(head))

    for left, right in pairs:
        names = disp_names(left, right, show_base)
        line = f"{short_name(left):<26}"
        for m in cfg.metrics:
            p = prec(cfg, m)
            for n in names:
                v = rows.get(n, {}).get(m)
                line += f" {'-':>12}" if v is None else f" {zf(v, f'>12.{p}f')}"
            line += f" {zf(rows[right][m] - rows[left][m], f'>+11.{p}f')}"
        print(line)

    # Prosecna razlika po metrici (right - left).
    print("-" * len(head))
    avg = f"{'PROSEK d':<26}"
    for m in cfg.metrics:
        deltas = [rows[r][m] - rows[l][m] for l, r in pairs]
        blank = " " * 12
        avg += " " + " ".join([blank] * len(labels))
        avg += f" {zf(sum(deltas) / len(deltas), f'>+11.{prec(cfg, m)}f')}"
    print(avg)
    print()
    print(avg_delta_line(cfg, rows, pairs, left_label, right_label, show_base))
    print()

    print("LaTeX tabela (za kopiranje):")
    print(build_table(cfg, rows, pairs, left_label, right_label, show_base))


def build_table(cfg, rows, pairs, left_label, right_label, show_base):
    """Vrati LaTeX tabelu (table[H] + tabular); bolduje najbolju vrednost po metrici."""
    labels = disp_labels(left_label, right_label, show_base)
    per = len(labels)  # kolona vrednosti po metrici (2 ili 3)
    ncol = len(cfg.metrics)

    # Grupisano zaglavlje: po `per` kolona za svaku metriku.
    top = ["Model"] + [r"\multicolumn{" + str(per) + r"}{c}{" + f"{m} {arrow(m)}" + "}"
                       for m in cfg.metrics]
    cmids = "".join(r"\cmidrule(lr){" + f"{2 + per * k}-{1 + per * (k + 1)}" + "}"
                    for k in range(ncol))
    sub = [""] + [" & ".join(tex_name(lab) for lab in labels)] * ncol

    def fmt(v, p, better):
        if v is None:
            return "-"
        s = zf(v, f".{p}f")
        return r"\textbf{" + s + "}" if better else s

    body = []
    for left, right in pairs:
        names = disp_names(left, right, show_base)
        cells = [tex_name(short_name(left))]
        for m in cfg.metrics:
            p = prec(cfg, m)
            best = best_val(rows, names, m)
            present = [rows[n][m] for n in names if n in rows]
            varies = len(set(present)) > 1
            for n in names:
                v = rows.get(n, {}).get(m)
                cells.append(fmt(v, p, v is not None and varies and v == best))
        body.append(tex_row(cells))

    table = tex_table("l" + ("c" * per) * ncol, [tex_row(top), cmids, tex_row(sub)], body,
                      cfg.caption or "Poređenje JPEG modela", cfg.label or f"tab:jpeg_{cfg.name}",
                      size=None, tabcolsep=None, centering=False, caption_below=True)
    return table + "\n\n" + avg_delta_tex(cfg, rows, pairs, left_label, right_label, show_base)


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
