"""
fp16_vs_fp32_quality_diff.py — koliko FP16 menja kvalitet u odnosu na FP32.

Uparuje modele po imenu izmedju FP32 i FP16 evaluacije istog skupa i za svaku
metriku (SSIM, LPIPS, PSNR) racuna apsolutnu razliku |FP16 - FP32| po modelu, pa
ispisuje minimalnu, prosecnu i maksimalnu apsolutnu razliku (uz prosecnu razliku
sa znakom, radi smera). Male vrednosti znace da poluprecinost ne kvari kvalitet.
"""

import csv
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.logger import Logger
from utils.path import get_results_path

DEFAULT_EXCLUDE = ["GAN", "ESRGAN", "jpeg"]
METRICS = ("SSIM", "PSNR", "LPIPS")
# Bazne interpolacije ne zavise od precinosti — izbacuju se iz poredjenja.
BASELINE_NAMES = ("nearest", "bilinear", "bicubic", "lanczos")

# Podrazumevane vrednosti za svako polje konfiguracije. Svaka stavka u CONFIGS
# prepisuje samo ono sto joj treba; ostalo se uzima odavde.
CONFIG_DEFAULTS = {
    "name": "default",  # koristi se za ime izlaznog fajla
    "fp32_results": "results_2x_Set14.csv",  # CSV sa FP32 kvalitetom
    "fp16_results": "results_2x_Set14_half.csv",  # CSV sa FP16 kvalitetom (_half)
    "metrics": list(METRICS),  # metrike za poredjenje (podskup METRICS)
    "archs": None,  # zadrzi samo ove arhitekture (npr. ["EDSR"])
    "include": None,  # zadrzi samo modele cije ime sadrzi neku nisku
    "exclude": None,  # None -> DEFAULT_EXCLUDE
    "out": None,  # None -> results/analysis/fp16_vs_fp32/..._<name>.txt
    "no_log": False,  # True -> samo ispis na ekran, bez fajla
}

##############################################


CONFIGS = [
    {
        "name": "Set14",
        "fp32_results": "results_2x_Set14.csv",
        "fp16_results": "results_2x_Set14_half.csv",
    },
]


##############################################


def resolve_results(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return get_results_path(name)


def arch_of(name: str) -> str:
    parts = name.split("_")
    if parts and parts[0] == "SR":
        parts = parts[1:]
    return parts[0] if parts else name


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def keep(name, exclude, archs, include):
    if not name or name in BASELINE_NAMES:
        return False
    if any(sub in name for sub in exclude):
        return False
    if include and not any(sub in name for sub in include):
        return False
    if archs and arch_of(name) not in archs:
        return False
    return True


def load_side(path: Path, metrics, exclude, archs, include):
    """Vrati {model_name: {metric: value, ...}}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [m for m in metrics if m not in reader.fieldnames]
        if missing:
            sys.exit(f"Metrike {', '.join(missing)} ne postoje u {path.name}.")
        for r in reader:
            name = r.get("model_name", "")
            if not keep(name, exclude, archs, include):
                continue
            rec = {}
            for m in metrics:
                v = to_float(r.get(m))
                if v is not None:
                    rec[m] = v
            if rec:
                out[name] = rec
    return out


def main():
    if not CONFIGS:
        sys.exit("CONFIGS je prazna — dodaj bar jednu konfiguraciju.")
    for i, cfg_dict in enumerate(CONFIGS):
        cfg = SimpleNamespace(**{**CONFIG_DEFAULTS, **cfg_dict})
        if i:
            print("\n" + "=" * 96 + "\n")
        print(f"### Konfiguracija: {cfg.name}\n")
        run_config(cfg)


def run_config(cfg):
    exclude = DEFAULT_EXCLUDE if cfg.exclude is None else cfg.exclude
    archs_filter = set(cfg.archs) if cfg.archs else None
    include = cfg.include or None

    fp32_path = resolve_results(cfg.fp32_results)
    fp16_path = resolve_results(cfg.fp16_results)
    for p in (fp32_path, fp16_path):
        if not p.exists():
            sys.exit(f"CSV ne postoji: {p}")

    fp32 = load_side(fp32_path, cfg.metrics, exclude, archs_filter, include)
    fp16 = load_side(fp16_path, cfg.metrics, exclude, archs_filter, include)
    common = sorted(set(fp32) & set(fp16))
    if not common:
        sys.exit("Nema modela prisutnih u obe precinosti (proveri fajlove/filtre).")

    # Sazetak po metrici: min/prosek/medijana/max apsolutne razlike + prosek sa znakom.
    summary = []  # (metric, n, min_abs, mean_abs, median_abs, max_abs, worst_model, mean_signed)
    for m in cfg.metrics:
        diffs = []  # (|razlika|, razlika_sa_znakom, ime)
        for name in common:
            a = fp16[name].get(m)
            b = fp32[name].get(m)
            if a is not None and b is not None:
                diffs.append((abs(a - b), a - b, name))
        if not diffs:
            continue
        abs_vals = [d for d, _, _ in diffs]
        signed = [s for _, s, _ in diffs]
        worst = max(diffs, key=lambda t: t[0])
        summary.append((m, len(diffs), min(abs_vals), sum(abs_vals) / len(abs_vals),
                        statistics.median(abs_vals), worst[0], worst[2],
                        sum(signed) / len(signed)))

    if not summary:
        sys.exit("Nema uparenih vrednosti ni za jednu metriku.")

    # Izlaz se opciono cuva u results/analysis/ preko Logger-a (isti kao u treningu).
    if cfg.no_log:
        report(summary, fp32_path, fp16_path, len(common))
    else:
        out_path = Path(cfg.out) if cfg.out else get_results_path(
            f"analysis/fp16_vs_fp32/fp16_vs_fp32_quality_diff_{cfg.name}.txt")
        with Logger(out_path):
            report(summary, fp32_path, fp16_path, len(common))


def report(summary, fp32_path, fp16_path, n_common):
    print(f"FP32 izvor: {fp32_path.name}")
    print(f"FP16 izvor: {fp16_path.name}")
    print(f"Uparenih modela: {n_common}  (razlika = FP16 - FP32)")
    print()
    header = (f"{'Metrika':<8} {'N':>4} {'min|d|':>12} {'prosek|d|':>12} "
              f"{'medijana|d|':>12} {'max|d|':>12} {'prosek d':>12}   najveca razlika")
    print(header)
    print("-" * len(header))
    for m, n, mn, mean_abs, med_abs, mx, worst_model, mean_signed in summary:
        print(f"{m:<8} {n:>4} {mn:>12.6f} {mean_abs:>12.6f} {med_abs:>12.6f} "
              f"{mx:>12.6f} {mean_signed:>+12.6f}   {worst_model}")


if __name__ == "__main__":
    main()
