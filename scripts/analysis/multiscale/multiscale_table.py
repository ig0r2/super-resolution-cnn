"""
multiscale_table.py - tabela: multi-scale naspram namenskih (single-scale) modela.

Za zadati skup (config) i vise faktora uvecanja, uparuje multi-scale model (bez
oznake faktora u imenu, npr. `RFDN_4_256`) sa namenskim single-scale modelom iste
konfiguracije (npr. `RFDN_2x_4_256`) na svakom faktoru. Ispisuje jednu sazetu
tabelu: jedan red po faktoru (×2/×3/×4), a za svaku metriku (PSNR/SSIM/LPIPS)
prosek namenskih naspram proseka multi-scale modela, boldujuci bolji prosek.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.analysis.config import run_configs
from utils.analysis.data import name_filter, read_rows, require_results, to_float
from utils.analysis.fmt import tex_row, tex_table, zf
from utils.analysis.metrics import arrow, check_metrics, higher_is_better, prec_for
from utils.analysis.names import INTERPOLATIONS, arch_of, pair_multiscale
from utils.path import get_results_path

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "Set14",  # koristi se za imena izlaznih fajlova
    "scales": [2, 3, 4],  # faktori (jedan red po faktoru u tabeli)
    "dataset": "Set14",  # skup — ulazi u ime CSV-a i naslov
    "template": "results_{scale}x_{dataset}_half.csv",  # sablon CSV-a po faktoru
    "metrics": ["PSNR", "SSIM", "LPIPS"],  # metrike (grupe kolona), redom
    "decimals": {"PSNR": 2, "SSIM": 4, "LPIPS": 4},  # broj decimala: None -> podrazumevano (4); int za sve;
    # ili dict npr. {"PSNR": 2, "SSIM": 4, "LPIPS": 3}
    "archs": None,  # zadrzi samo ove arhitekture (npr. ["RFDN"])
    "include": None,  # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": ["GAN", "ESRGAN", "jpeg"],  # niske koje se izbacuju
    "show_n": False,  # True -> kolona "$n$" (broj uparenih konfiguracija)
    "caption": None,  # None -> automatski
    "label": None,  # None -> tab:multiscale_{dataset}
    "tex_out": None,  # None -> ..._<name>.tex
    "no_log": False,  # True -> samo ispis na ekran, bez .tex fajla
}

##############################################


CONFIGS = [
    {
        "name": "DIV2K",
        "dataset": "DIV2K",
        "scales": [2, 3, 4],
    },
]


##############################################


def load_pairs(path, cfg):
    """Vrati listu parova (key, single{metric:v}, multi{metric:v}) sa svim metrikama.

    Uparuje multi-scale (bez tokena faktora) i single-scale (sa tokenom) model
    iste konfiguracije; zadrzava samo one koji imaju sve trazene metrike.
    """
    keep = name_filter(cfg.exclude or [], cfg.archs, cfg.include)
    items = []
    for r in read_rows(path, cfg.metrics):
        name = r.get("model_name", "")
        if name in INTERPOLATIONS or not keep(name):
            continue
        rec = {m: to_float(r.get(m)) for m in cfg.metrics}
        if all(v is not None for v in rec.values()):
            items.append((name, rec))
    pairs = [(key, s, mu) for key, (_, s), (_, mu) in pair_multiscale(items)]
    # Sortiraj po arhitekturi pa po imenu konfiguracije.
    pairs.sort(key=lambda t: (arch_of(t[0]), t[0]))
    return pairs


def run_config(cfg):
    check_metrics(cfg.metrics)
    if not cfg.scales:
        sys.exit("Zadaj bar jedan faktor u 'scales' (npr. [2, 3, 4]).")

    # Za svaki faktor: prosek namenskih i multi-scale po metrici + broj parova.
    rows = []  # (scale, n, {metric: (s_mean, mu_mean, wins)})
    for s in cfg.scales:
        path = require_results(cfg.template.format(scale=s, dataset=cfg.dataset))
        pairs = load_pairs(path, cfg)
        if not pairs:
            print(f"Upozorenje: nema uparenih konfiguracija u {path.name} — preskacem x{s}.")
            continue
        rows.append((s, len(pairs), means_for(cfg, pairs)))
    if not rows:
        sys.exit("Nema uparenih multi-scale / single-scale konfiguracija ni za jedan faktor.")

    report(cfg, rows)
    if not cfg.no_log:
        tex_path = Path(cfg.tex_out) if cfg.tex_out else get_results_path(
            f"analysis/multiscale/multiscale_table_{cfg.name}.tex")
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(build_table(cfg, rows), encoding="utf-8")
        print(f"LaTeX tabela: {tex_path}")


def means_for(cfg, pairs):
    """Za svaku metriku: (prosek namenskih, prosek multi, broj multi-pobeda)."""
    stats = {}
    for m in cfg.metrics:
        higher = higher_is_better(m)
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

    head = f"{'Faktor':<8}" + (f"{'n':>5}" if cfg.show_n else "")
    for m in cfg.metrics:
        head += f" {m + ' nam.':>12} {m + ' multi':>12} {'d' + m:>11}"
    print(head)
    print("-" * len(head))

    for s, n, stats in rows:
        line = f"{'x' + str(s):<8}" + (f"{n:>5}" if cfg.show_n else "")
        for m in cfg.metrics:
            prec = prec_for(cfg.decimals, m)
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
    print(build_table(cfg, rows))


def build_table(cfg, rows):
    """Vrati sazetu LaTeX (booktabs) tabelu: red po faktoru, prosek nam./multi."""
    scales_txt = ", ".join(f"$\\times{s}$" for s, _, _ in rows)
    caption = cfg.caption or (
        f"Prosečan kvalitet namenskih (single-scale) i multi-scale modela iste "
        f"konfiguracije na skupu {cfg.dataset}, po faktoru uvećanja ({scales_txt}). "
        f"Boldovan je bolji prosek u paru; strelice označavaju poželjan smer metrike.")
    label = cfg.label or f"tab:multiscale_{cfg.dataset}"

    def fmt(v, prec, better):
        s = zf(v, f".{prec}f")
        return r"\textbf{" + s + "}" if better else s

    ncol = len(cfg.metrics)
    # Grupisano zaglavlje: po dve kolone (namenski / multi) za svaku metriku.
    top = ["Faktor"] + (["$n$"] if cfg.show_n else [])
    top += [r"\multicolumn{2}{c}{" + f"{m} {arrow(m)}" + "}" for m in cfg.metrics]
    base = 2 + (1 if cfg.show_n else 0)  # prva kolona metrike
    cmids = "".join(r"\cmidrule(lr){" + f"{base + 2 * k}-{base + 1 + 2 * k}" + "}"
                    for k in range(ncol))
    sub = [""] + ([""] if cfg.show_n else []) + ["namenski & multi"] * ncol

    body = []
    for s, n, stats in rows:
        cells = [f"$\\times{s}$"] + ([str(n)] if cfg.show_n else [])
        for m in cfg.metrics:
            prec = prec_for(cfg.decimals, m)
            s_mean, mu_mean, _ = stats[m]
            multi_better = (mu_mean > s_mean) if higher_is_better(m) else (mu_mean < s_mean)
            equal = mu_mean == s_mean
            cells.append(fmt(s_mean, prec, not multi_better and not equal))
            cells.append(fmt(mu_mean, prec, multi_better and not equal))
        body.append(tex_row(cells))

    colspec = "@{}l" + ("r" if cfg.show_n else "") + "cc" * ncol + "@{}"
    return tex_table(colspec, [tex_row(top), cmids, tex_row(sub)], body, caption, label,
                     group=True, tabcolsep="5pt", arraystretch="1.15")


if __name__ == "__main__":
    run_configs(CONFIGS, CONFIG_DEFAULTS, run_config)
