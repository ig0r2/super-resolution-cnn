import copy
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from utils.checkpoints import load_model_from_checkpoint
from utils.csv_utils import save_to_csv, is_model_evaluated
from utils.path import get_logs_path, get_results_path, get_checkpoints_path
from utils.video.evaluator_perf_video import EvaluatorPerfVideo, Runtype
from utils.logger import Logger

# Evaluacija brzine modela za OpenCV video player demo

if __name__ == "__main__":
    UPSCALE_FACTOR: Literal[2, 3, 4] = 2
    RUNTYPE: Runtype = "tensorrt"

    USE_FP32 = False

    EVALUATE_PERFORMANCE_720p = True
    EVALUATE_PERFORMANCE_480p = True

    EVALUATE_TILING_128 = True
    EVALUATE_TILING_256 = True

    SKIP_EVALUATED = True

    CHECKPOINT_PATHS = [
        # get_checkpoints_path("multiscale/SR_FastEDSR_4_256.pth"),
    ]

    #####################################################
    if len(CHECKPOINT_PATHS) == 0:
        CHECKPOINT_PATHS = sorted(p for pattern in (f"{UPSCALE_FACTOR}x/*.pth", "multiscale/*.pth")
                                  for p in get_checkpoints_path().glob(pattern)
                                  if not p.name.endswith("_latest.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    csv_path = get_results_path(f"results_{UPSCALE_FACTOR}x_FPS{'_fp32' if USE_FP32 else ''}.csv")

    for checkpoint_path in CHECKPOINT_PATHS:
        checkpoint_path = Path(checkpoint_path)
        model_name = checkpoint_path.stem

        if SKIP_EVALUATED and is_model_evaluated(csv_path, model_name):
            print(f"Skipping {model_name}")
            continue

        log_path = get_logs_path(f"evaluation/eval_{checkpoint_path.stem}.txt")

        with (Logger(log_path)):
            print("=" * 50)
            print(f"Checkpoint: {checkpoint_path.stem}")

            model, model_conf = load_model_from_checkpoint(checkpoint_path, device)
            model.upscale_factor = UPSCALE_FACTOR
            name = model_conf['checkpoint_name']

            # Parameter number
            total_params = sum(p.numel() for p in model.parameters())
            print("-" * 30)
            print(f"Total parameters: {total_params:,}")
            print("-" * 30)

            # Performance
            perf_results = {'720p': "", '480p': "", '720p (128x128)': "", '480p (128x128)': "", '720p (256x256)': "",
                            '480p (256x256)': ""}
            if EVALUATE_PERFORMANCE_720p:
                perf_results['720p'] = (
                    EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(720, 1280),
                                       upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE,
                                           use_fp32=USE_FP32).evaluate())
            if EVALUATE_PERFORMANCE_480p:
                perf_results['480p'] = (
                    EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(480, 854),
                                       upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE,
                                           use_fp32=USE_FP32).evaluate())
            if EVALUATE_TILING_128:
                if EVALUATE_PERFORMANCE_720p:
                    perf_results['720p (128x128)'] = (
                        EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(720, 1280), tiled=True,
                                           tile_size=128, upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE,
                                           use_fp32=USE_FP32).evaluate())
                if EVALUATE_PERFORMANCE_480p:
                    perf_results['480p (128x128)'] = (
                        EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(480, 854), tiled=True,
                                           tile_size=128, upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE,
                                           use_fp32=USE_FP32).evaluate())
            if EVALUATE_TILING_256:
                if EVALUATE_PERFORMANCE_720p:
                    perf_results['720p (256x256)'] = (
                        EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(720, 1280), tiled=True,
                                           tile_size=256, upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE,
                                           use_fp32=USE_FP32).evaluate())
                if EVALUATE_PERFORMANCE_480p:
                    perf_results['480p (256x256)'] = (
                        EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(480, 854), tiled=True,
                                           tile_size=256, upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE,
                                           use_fp32=USE_FP32).evaluate())

            # Save to CSV
            save_to_csv({
                "model_name": checkpoint_path.stem,
                "params": total_params,
                "runtype": RUNTYPE,
                **perf_results,
            }, csv_path)
