import sys
from pathlib import Path
from typing import Literal

# torch.onnx prints unicode (✅) on export; keep it from crashing a cp1252 console/logger.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Run as a script: put the project root on sys.path so `videoplayer_nvdec` / `utils` import.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from utils.checkpoints import load_model_from_checkpoint  # noqa: E402
from utils.csv_utils import save_to_csv  # noqa: E402
from utils.path import get_logs_path, get_results_path, get_checkpoints_path, get_project_root  # noqa: E402
from utils.logger import Logger  # noqa: E402
from videoplayer_nvdec.evaluator_perf_video_nvdec import EvaluatorPerfVideoNVDEC  # noqa: E402

# Speed evaluation for the NVDEC-decode video player pipeline. One CSV row per model (like
# scripts/evaluation/evaluate_script_video.py), with per-resolution, per-stage columns:
# "<label> decode/sr/e2e/full". The row is upserted after every resolution, so results persist
# incrementally and an interrupted run resumes where it left off.

STAGES = ("decode", "sr", "e2e", "full")

if __name__ == "__main__":
    UPSCALE_FACTOR: Literal[2, 3, 4] = 2
    RUNTYPE = "tensorrt-nvdec"

    SKIP_EVALUATED = True

    WARMUP_RUNS = 10
    ITERATIONS = 100

    VIDEOS = [
        ("480p", get_project_root("videoinput/ldv-001-480p.mp4")),
    ]

    CHECKPOINT_PATHS = [
        # get_checkpoints_path("multiscale/SR_FastEDSR_4_32.pth"),
    ]

    #####################################################
    if len(CHECKPOINT_PATHS) == 0:
        CHECKPOINT_PATHS = sorted(p for pattern in (f"{UPSCALE_FACTOR}x/*.pth", "multiscale/*.pth")
                                  for p in get_checkpoints_path().glob(pattern)
                                  if not p.name.endswith("_latest.pth"))

    csv_path = get_results_path(f"results_{UPSCALE_FACTOR}x_nvdec_FPS.csv")

    expected_cols = [f"{label} {stage}" for label, _ in VIDEOS for stage in STAGES]


    def already_done(model_name):
        """True only if the model's row exists and every expected column is filled in."""
        if not Path(csv_path).exists():
            return False
        df = pd.read_csv(csv_path, dtype=str)
        row = df[(df.get("model_name") == model_name) & (df.get("runtype") == RUNTYPE)]
        if row.empty:
            return False
        row = row.iloc[0]
        return all(c in row and str(row[c]) not in ("", "nan") for c in expected_cols)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    for checkpoint_path in CHECKPOINT_PATHS:
        checkpoint_path = Path(checkpoint_path)
        model_name = checkpoint_path.stem

        if SKIP_EVALUATED and already_done(model_name):
            print(f"Skipping {model_name}")
            continue

        log_path = get_logs_path(f"evaluation/eval_nvdec_{checkpoint_path.stem}.txt")

        with Logger(log_path):
            print("=" * 50)
            print(f"Checkpoint: {checkpoint_path.stem}")

            # Load once for param count + config name; TRTBackendNVDEC reloads the checkpoint itself.
            model, model_conf = load_model_from_checkpoint(checkpoint_path, "cpu")
            model.upscale_factor = UPSCALE_FACTOR
            name = model_conf['checkpoint_name']
            total_params = sum(p.numel() for p in model.parameters())
            del model
            print("-" * 30)
            print(f"Total parameters: {total_params:,}")
            print("-" * 30)

            row = {"model_name": checkpoint_path.stem, "params": total_params, "runtype": RUNTYPE}

            for label, video_path in VIDEOS:
                video_path = Path(video_path)
                if not video_path.exists():
                    print(f"Video missing, skipping: {video_path.name}")
                    continue

                print(f"\n>>> {model_name} @ {label} ({video_path.name})")
                try:
                    metrics = EvaluatorPerfVideoNVDEC(
                        checkpoint_path=checkpoint_path, name=name, video_path=video_path,
                        upscale_factor=UPSCALE_FACTOR, warmup_runs=WARMUP_RUNS,
                        iterations=ITERATIONS).evaluate()
                except RuntimeError as e:
                    print(f"Skipping ({name} @ {label}): {e}")
                    torch.cuda.empty_cache()
                    metrics = {stage: "" for stage in STAGES}

                for stage in STAGES:
                    row[f"{label} {stage}"] = metrics[stage]

                # Upsert the model's row after every resolution (save after each model, and then some).
                save_to_csv(row, csv_path, match_keys=("model_name", "runtype"))
