"""
multiscale_table.py - tabela: multi-scale naspram namenskih (single-scale) modela.

Za zadati skup (config) i vise faktora uvecanja, uparuje multi-scale model (bez
oznake faktora u imenu, npr. `RFDN_4_256`) sa namenskim single-scale modelom iste
konfiguracije (npr. `RFDN_2x_4_256`) na svakom faktoru. Ispisuje jednu sazetu
tabelu: jedan red po faktoru (×2/×3/×4), a za svaku metriku (PSNR/SSIM/LPIPS)
prosek namenskih naspram proseka multi-scale modela, boldujuci bolji prosek.
"""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.logger import Logger
from utils.path import get_results_path

# Konzola je cesto cp1252; caption/labele imaju ne-ASCII pa reconfigure da print ne puca.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Metrika -> (broj decimala, da li je vece bolje, strelica za LaTeX header).
METRIC_INFO = {
    "PSNR": (4, True, r"$\uparrow$"),
    "SSIM": (4, True, r"$\uparrow$"),
    "LPIPS": (4, False, r"$\downarrow$"),
}

# Bazne interpolacije nisu modeli — izbacuju se iz poredjenja.
BASELINE_NAMES = ("nearest", "bilinear", "bicubic", "lanczos")

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "Set14",  # koristi se za imena izlaznih fajlova
    "scales": [2, 3, 4],  # faktori (jedan red po faktoru u tabeli)
    "dataset": "Set14",  # skup — ulazi u ime CSV-a i naslov
    "template": "results_{scale}x_{dataset}_half.csv",  # sablon CSV-a po faktoru
    "metrics": ["PSNR", "SSIM", "LPIPS"],  # metrike (grupe kolona), redom
    "decimals": 4,  # decimale: None -> podrazumevano (4); int za sve;
    # ili dict npr. {"PSNR": 2, "SSIM": 4, "LPIPS": 4}
    "archs": None,  # zadrzi samo ove arhitekture (npr. ["RFDN"])
    "include": None,  # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": ["GAN", "ESRGAN", "jpeg"],  # niske koje se izbacuju
    "show_n": False,  # True -> kolona "$n$" (broj uparenih konfiguracija)
    "caption": None,  # None -> automatski
    "label": None,  # None -> tab:multiscale_{dataset}
    "out": None,  # None -> ..._<name>.txt
    "tex_out": None,  # None -> ..._<name>.tex
    "no_log": False,  # True -> samo ispis na ekran, bez fajlova
}

##############################################


CONFIGS = [
    {
        "name": "Set14",
        "dataset": "Set14",
        "scales": [2, 3, 4],
    },
    {
        "name": "DIV2K",
        "dataset": "DIV2K",
        "scales": [2, 3, 4],
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


def is_single_scale(name: str) -> bool:
    return any(t.endswith("x") and t[:-1].isdigit() for t in name.split("_"))


def config_key(name: str) -> str:
    """Ime bez `SR_` prefiksa i bez tokena faktora — kljuc za uparivanje."""
    parts = [t for t in name.split("_") if t != "SR"
             and not (t.endswith("x") and t[:-1].isdigit())]
    return "_".join(parts)


def display_name(key: str) -> str:
    """config_key 'RFDN_4_256' -> 'RFDN 4/256' (bez _r tokena)."""
    parts = [p for p in key.split("_") if p != "r"]
    if not parts:
        return key
    arch, rest = parts[0], parts[1:]
    return f"{arch} {'/'.join(rest)}" if rest else arch


def prec_for(cfg, metric):
    """Broj decimala za metriku prema cfg.decimals (None/int/dict)."""
    d = getattr(cfg, "decimals", None)
    if d is None:
        return METRIC_INFO[metric][0]
    if isinstance(d, dict):
        return d.get(metric, METRIC_INFO[metric][0])
    return d


def load_pairs(path, metrics, exclude, archs_filter, include):
    """Vrati listu parova (key, single{metric:v}, multi{metric:v}) sa svim metrikama.

    Uparuje multi-scale (bez tokena faktora) i single-scale (sa tokenom) model
    iste konfiguracije; zadrzava samo one koji imaju sve trazene metrike.
    """
    multi, single = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [m for m in metrics if m not in reader.fieldnames]
        if missing:
            sys.exit(f"Metrike {', '.join(missing)} ne postoje u {path.name}.")
        for r in reader:
            name = r.get("model_name", "")
            if not name or name in BASELINE_NAMES:
                continue
            if any(sub in name for sub in exclude):
                continue
            if include and not any(sub in name for sub in include):
                continue
            if archs_filter and arch_of(name) not in archs_filter:
                continue
            rec = {}
            for m in metrics:
                v = to_float(r.get(m))
                if v is not None:
                    rec[m] = v
            if len(rec) != len(metrics):
                continue
            (single if is_single_scale(name) else multi)[config_key(name)] = rec

    pairs = []
    for key in set(multi) & set(single):
        pairs.append((key, single[key], multi[key]))
    # Sortiraj po arhitekturi pa po imenu konfiguracije.
    pairs.sort(key=lambda t: (arch_of(t[0]), t[0]))
    return pairs


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
    if not cfg.scales:
        sys.exit("Zadaj bar jedan faktor u 'scales' (npr. [2, 3, 4]).")

    exclude = cfg.exclude or []
    archs_filter = set(cfg.archs) if cfg.archs else None
    include = cfg.include or None

    # Za svaki faktor: prosek namenskih i multi-scale po metrici + broj parova.
    rows = []  # (scale, n, {metric: (s_mean, mu_mean, wins)})
    for s in cfg.scales:
        path = resolve_results(cfg.template.format(scale=s, dataset=cfg.dataset))
        if not path.exists():
            sys.exit(f"CSV za faktor x{s} ne postoji: {path}")
        pairs = load_pairs(path, cfg.metrics, exclude, archs_filter, include)
        if not pairs:
            print(f"Upozorenje: nema uparenih konfiguracija u {path.name} — preskacem x{s}.")
            continue
        rows.append((s, len(pairs), means_for(cfg, pairs)))
    if not rows:
        sys.exit("Nema uparenih multi-scale / single-scale konfiguracija ni za jedan faktor.")

    if cfg.no_log:
        report(cfg, rows)
    else:
        txt_path = Path(cfg.out) if cfg.out else get_results_path(
            f"analysis/multiscale/multiscale_table_{cfg.name}.txt")
        with Logger(txt_path):
            report(cfg, rows)
        tex_path = Path(cfg.tex_out) if cfg.tex_out else get_results_path(
            f"analysis/multiscale/multiscale_table_{cfg.name}.tex")
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(tex_table(cfg, rows), encoding="utf-8")
        print(f"LaTeX tabela: {tex_path}")


def means_for(cfg, pairs):
    """Za svaku metriku: (prosek namenskih, prosek multi, broj multi-pobeda)."""
    stats = {}
    for m in cfg.metrics:
        _, higher, _ = METRIC_INFO[m]
        s_vals = [s[m] for _, s, _ in pairs]
        mu_vals = [mu[m] for _, _, mu in pairs]
        wins = sum(1 for s, mu in zip(s_vals, mu_vals)
                   if (mu > s if higher else mu < s))
        stats[m] = (sum(s_vals) / len(s_vals), sum(mu_vals) / len(mu_vals), wins)
    return stats


def report(cfg, rows):
    print(f"Skup: {cfg.dataset}   faktori: {', '.join('x' + str(s) for s, _, _ in rows)}")
    print("Prosek namenskih (single-scale) naspram proseka multi-scale modela.")
    print()

    show_n = getattr(cfg, "show_n", True)
    head = f"{'Faktor':<8}" + (f"{'n':>5}" if show_n else "")
    for m in cfg.metrics:
        head += f" {m + ' nam.':>12} {m + ' multi':>12} {'d' + m:>11}"
    print(head)
    print("-" * len(head))

    for s, n, stats in rows:
        line = f"{'x' + str(s):<8}" + (f"{n:>5}" if show_n else "")
        for m in cfg.metrics:
            prec = prec_for(cfg, m)
            s_mean, mu_mean, _ = stats[m]
            line += (f" {zf(s_mean, f'>12.{prec}f')} {zf(mu_mean, f'>12.{prec}f')}"
                     f" {zf(mu_mean - s_mean, f'>+11.{prec}f')}")
        print(line)
    print()

    for s, n, stats in rows:
        summary = "   ".join(f"{m}: multi bolji {stats[m][2]}/{n}" for m in cfg.metrics)
        print(f"x{s} pobede -> {summary}")
    print()

    print("LaTeX tabela (za kopiranje):")
    print(tex_table(cfg, rows))


def tex_table(cfg, rows):
    """Vrati sazetu LaTeX (booktabs) tabelu: red po faktoru, prosek nam./multi."""
    scales_txt = ", ".join(f"$\\times{s}$" for s, _, _ in rows)
    caption = cfg.caption or (
        f"Prosečan kvalitet namenskih (single-scale) i multi-scale modela iste "
        f"konfiguracije na skupu {cfg.dataset}, po faktoru uvećanja ({scales_txt}). "
        f"Boldovan je bolji prosek u paru; strelice označavaju poželjan smer metrike.")
    label = cfg.label or f"tab:multiscale_{cfg.dataset}"

    show_n = getattr(cfg, "show_n", True)

    def fmt(v, prec, better):
        s = zf(v, f".{prec}f")
        return r"\textbf{" + s + "}" if better else s

    ncol = len(cfg.metrics)
    colspec = "@{}l" + ("r" if show_n else "") + "cc" * ncol + "@{}"
    lines = [r"\begin{table}[H]", r"    \centering",
             f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
             r"    \begingroup", r"    \small",
             r"    \setlength{\tabcolsep}{5pt}",
             r"    \renewcommand{\arraystretch}{1.15}",
             f"    \\begin{{tabular}}{{{colspec}}}",
             r"        \toprule"]

    # Grupisano zaglavlje: po dve kolone (namenski / multi) za svaku metriku.
    top = ["Faktor"] + (["$n$"] if show_n else [])
    for m in cfg.metrics:
        _, _, arrow = METRIC_INFO[m]
        top.append(r"\multicolumn{2}{c}{" + f"{m} {arrow}" + "}")
    lines.append("        " + " & ".join(top) + r" \\")
    base = 2 + (1 if show_n else 0)  # prva kolona metrike
    cmids = "".join(r"\cmidrule(lr){" + f"{base + 2 * k}-{base + 1 + 2 * k}" + "}"
                    for k in range(ncol))
    lines.append("        " + cmids)
    sub = [""] + ([""] if show_n else []) + ["namenski & multi"] * ncol
    lines.append("        " + " & ".join(sub) + r" \\")
    lines.append(r"        \midrule")

    for s, n, stats in rows:
        cells = [f"$\\times{s}$"] + ([str(n)] if show_n else [])
        for m in cfg.metrics:
            prec = prec_for(cfg, m)
            _, higher, _ = METRIC_INFO[m]
            s_mean, mu_mean, _ = stats[m]
            multi_better = (mu_mean > s_mean) if higher else (mu_mean < s_mean)
            equal = mu_mean == s_mean
            cells.append(fmt(s_mean, prec, not multi_better and not equal))
            cells.append(fmt(mu_mean, prec, multi_better and not equal))
        lines.append("        " + " & ".join(cells) + r" \\")

    lines += [r"        \bottomrule", r"    \end{tabular}",
              r"    \endgroup", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
