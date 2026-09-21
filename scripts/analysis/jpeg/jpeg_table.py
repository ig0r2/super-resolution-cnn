"""
jpeg_table.py - poredjenje baznog modela i njegovog jpeg parnjaka.

Za svaki model koji ima jpeg parnjaka (npr. SR_FastEDSR_4_64 <-> SR_FastEDSR_
jpeg_4_64) u zadatom CSV-u, ispisuje tabelu sa metrikama baznog i jpeg modela i
njihovom razlikom (Δ = jpeg - bazni). Uz obicnu tabelu, izbacuje i gotovu LaTeX
tabelu koja se moze direktno prekopirati u tex dokument (bolduje bolju vrednost
po metrici).
"""

import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path import get_results_path

# Konzola je cesto cp1252; ispis .tex sadrzi ć/č pa reconfigure da print ne puca.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Metrike: (broj decimala, da li je vece bolje, strelica za LaTeX header).
METRIC_INFO = {
    "LPIPS": (4, False, r"$\downarrow$"),
    "SSIM": (4, True, r"$\uparrow$"),
    "PSNR": (2, True, r"$\uparrow$"),
}

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
        "name": "Set14_jpeg",
        "results": "results_2x_Set14_jpeg_half.csv",
        "pairing": "base_vs_jpeg",
    },
    {
        "name": "Set14_jpeg_vs_s",
        "metrics": ["SSIM", "LPIPS"],
        "results": "results_2x_Set14_jpeg_half.csv",
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
                params[name] = to_float(r.get("params"))
    return out, params


def short_name(name: str) -> str:
    """Ime za prikaz: 'SR_FastEDSR_2_64' -> 'FastEDSR 2/64' (bez jpeg/scale/_r)."""
    parts = [p for p in name.split("_") if p != "jpeg"]
    if parts and parts[0] == "SR":
        parts = parts[1:]
    parts = [p for p in parts if not re.fullmatch(r"\d+x", p) and p != "r"]
    if not parts:
        return name
    arch, rest = parts[0], parts[1:]
    return f"{arch} {'/'.join(rest)}" if rest else arch


def base_name(name: str) -> str:
    """jpeg model -> bazni parnjak (bez 'jpeg' tokena), zadrzava SR_ prefiks."""
    return "_".join(p for p in name.split("_") if p != "jpeg")


def disp_labels(left_label, right_label, show_base):
    """Labele kolona vrednosti (redom); opciono i 'bazni' na pocetku."""
    return (["bazni"] if show_base else []) + [left_label, right_label]


def disp_names(left_name, right_name, show_base):
    """Imena modela za kolone vrednosti (redom), uparena sa disp_labels."""
    return ([base_name(left_name)] if show_base else []) + [left_name, right_name]


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
    # Bazna kolona ima smisla samo kad su levi modeli jpeg (pairing jpeg_vs_s).
    show_base = getattr(cfg, "show_base", False) and cfg.pairing == "jpeg_vs_s"

    results_path = resolve_results(cfg.results)
    if not results_path.exists():
        sys.exit(f"CSV ne postoji: {results_path}")

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
    tex_path.write_text(tex_table(cfg, rows, pairs, left_label, right_label, show_base),
                        encoding="utf-8")
    print(f"LaTeX tabela: {tex_path}")


def prec_for(cfg, metric):
    """Broj decimala za metriku prema cfg.decimals (None/int/dict)."""
    d = getattr(cfg, "decimals", None)
    if d is None:
        return METRIC_INFO[metric][0]
    if isinstance(d, dict):
        return d.get(metric, METRIC_INFO[metric][0])
    return d


def best_val(rows, names, m):
    """Najbolja vrednost metrike m medju prisutnim modelima iz `names`."""
    _, higher, _ = METRIC_INFO[m]
    present = [rows[n][m] for n in names if n in rows]
    if not present:
        return None
    return (max if higher else min)(present)


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
    return ",  ".join(f"{m} {zf(avg[m], f'+.{prec_for(cfg, m)}f')}"
                      for m in cfg.metrics if avg[m] is not None)


def _delta_items_tex(cfg, avg):
    items = []
    for m in cfg.metrics:
        if avg[m] is None:
            continue
        sign = "-" if avg[m] < 0 else "+"
        items.append(f"{m} ${sign}${zf(abs(avg[m]), f'.{prec_for(cfg, m)}f')}")
    return ", ".join(items)


def avg_delta_line(cfg, rows, pairs, left_label, right_label, show_base):
    """Tekstualne recenice sa prosecnom razlikom po metrici (za konzolu)."""
    out = [f"Prosecna razlika ({right_label} - {left_label}) po metrici: "
           + _delta_items_plain(cfg, avg_delta_map(
        rows, pairs, cfg.metrics, lambda l, r: r, lambda l, r: l))]
    if show_base:
        out.append(f"Prosecna razlika ({right_label} - bazni) po metrici: "
                   + _delta_items_plain(cfg, avg_delta_map(
            rows, pairs, cfg.metrics, lambda l, r: r,
            lambda l, r: base_name(l))))
    return "\n".join(out)


def avg_delta_tex(cfg, rows, pairs, left_label, right_label, show_base):
    """LaTeX recenice sa prosecnom razlikom po metrici (ispod tabele)."""
    esc = lambda s: s.replace("_", r"\_")
    out = [r"\noindent Prosečna razlika (" + esc(right_label) + r" $-$ "
           + esc(left_label) + "): " + _delta_items_tex(cfg, avg_delta_map(
        rows, pairs, cfg.metrics, lambda l, r: r, lambda l, r: l)) + "."]
    if show_base:
        out.append(r"\noindent Prosečna razlika (" + esc(right_label) + r" $-$ bazni): "
                   + _delta_items_tex(cfg, avg_delta_map(
            rows, pairs, cfg.metrics, lambda l, r: r,
            lambda l, r: base_name(l))) + ".")
    return "\n\n".join(out)


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
            prec = prec_for(cfg, m)
            for n in names:
                v = rows.get(n, {}).get(m)
                line += f" {'-':>12}" if v is None else f" {zf(v, f'>12.{prec}f')}"
            line += f" {zf(rows[right][m] - rows[left][m], f'>+11.{prec}f')}"
        print(line)

    # Prosecna razlika po metrici (right - left).
    print("-" * len(head))
    avg = f"{'PROSEK d':<26}"
    for m in cfg.metrics:
        prec = prec_for(cfg, m)
        deltas = [rows[r][m] - rows[l][m] for l, r in pairs]
        blank = " " * 12
        avg += " " + " ".join([blank] * len(labels))
        avg += f" {zf(sum(deltas) / len(deltas), f'>+11.{prec}f')}"
    print(avg)
    print()
    print(avg_delta_line(cfg, rows, pairs, left_label, right_label, show_base))
    print()

    print("LaTeX tabela (za kopiranje):")
    print(tex_table(cfg, rows, pairs, left_label, right_label, show_base))


def tex_table(cfg, rows, pairs, left_label, right_label, show_base):
    """Vrati LaTeX tabelu (table[H] + tabular); bolduje najbolju vrednost po metrici."""
    labels = disp_labels(left_label, right_label, show_base)
    per = len(labels)  # kolona vrednosti po metrici (2 ili 3)
    ncol = len(cfg.metrics)
    caption = cfg.caption or "Poređenje JPEG modela"
    label = cfg.label or f"tab:jpeg_{cfg.name}"

    # Grupisano zaglavlje: po `per` kolona za svaku metriku.
    top = ["Model"]
    for m in cfg.metrics:
        _, _, arrow = METRIC_INFO[m]
        top.append(r"\multicolumn{" + str(per) + r"}{c}{" + f"{m} {arrow}" + "}")
    cmids = "".join(r"\cmidrule(lr){" + f"{2 + per * k}-{1 + per * (k + 1)}" + "}"
                    for k in range(ncol))
    sub_labels = " & ".join(lab.replace("_", r"\_") for lab in labels)
    sub = [""] + [sub_labels] * ncol

    def fmt(v, prec, better):
        if v is None:
            return "-"
        s = zf(v, f".{prec}f")
        return r"\textbf{" + s + "}" if better else s

    lines = [r"\begin{table}[H]",
             r"    \begin{tabular}{l" + ("c" * per) * ncol + "}",
             r"        \toprule",
             "        " + " & ".join(top) + r" \\",
             "        " + cmids,
             "        " + " & ".join(sub) + r" \\",
             r"        \midrule"]

    for left, right in pairs:
        names = disp_names(left, right, show_base)
        cells = [short_name(left).replace("_", r"\_")]
        for m in cfg.metrics:
            prec = prec_for(cfg, m)
            best = best_val(rows, names, m)
            present = [rows[n][m] for n in names if n in rows]
            varies = len(set(present)) > 1
            for n in names:
                v = rows.get(n, {}).get(m)
                cells.append(fmt(v, prec, v is not None and varies and v == best))
        lines.append("        " + " & ".join(cells) + r" \\")

    lines += [r"        \bottomrule",
              r"    \end{tabular}",
              f"    \\caption{{{caption}}}",
              f"    \\label{{{label}}}",
              r"\end{table}",
              "",
              avg_delta_tex(cfg, rows, pairs, left_label, right_label, show_base)]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
