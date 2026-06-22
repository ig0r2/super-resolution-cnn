import copy
from pathlib import Path
from typing import Literal

import torch

from models import RegularModel
from utils.checkpoints import load_model_from_checkpoint
from utils.csv_utils import save_to_csv, get_columns_to_evaluate
from utils.evaluator import Evaluator
from utils.evaluator_perf import EvaluatorPerf
from datasets import get_test_set
from utils.logger import Logger

# Evaluacija modela i/ili metoda

from utils.path import get_results_path, get_logs_path, get_checkpoints_path

if __name__ == "__main__":
    UPSCALE_FACTOR: Literal[1, 2, 3, 4] = 2
    TEST_SET: Literal["DIV2K", "Set5", "Set14", "BSD100", "Urban100"] = "DIV2K"

    USE_HALF = True
    EVALUATE_METRICS = True
    JPEG_DEGRADATION = False

    EVALUATE_PERFORMANCE_720p = False
    EVALUATE_PERFORMANCE_480p = False
    USE_TENSORRT = True

    CHECKPOINT_PATHS = [
        # get_checkpoints_path("multiscale/SR_FastEDSR_4_128.pth"),
    ]

    METHODS = [
        # 'nearest',
        # 'bilinear',
        # 'bicubic',
        # 'lanczos'
    ]

    #####################################################
    if len(CHECKPOINT_PATHS) == 0:
        CHECKPOINT_PATHS = sorted(p for pattern in (f"{UPSCALE_FACTOR}x/*.pth", "multiscale/*.pth")
                                  for p in get_checkpoints_path().glob(pattern)
                                  if not p.name.endswith("_latest.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if EVALUATE_METRICS:
        test_set = get_test_set(name=TEST_SET, upscale_factor=UPSCALE_FACTOR, preload=len(CHECKPOINT_PATHS) > 1,
                                normalize=False, jpeg_degradation=JPEG_DEGRADATION)
        evaluator = Evaluator(test_set=test_set, device=device, use_half=USE_HALF)

    # Combine checkpoints and methods
    items_to_evaluate = [{'path': p, 'is_method': False} for p in CHECKPOINT_PATHS]
    items_to_evaluate += [{'method': m, 'is_method': True} for m in METHODS]

    csv_path = get_results_path(
        f"results_{UPSCALE_FACTOR}x_{TEST_SET}{'_jpeg' if JPEG_DEGRADATION else ''}{'_half' if USE_HALF else ''}.csv")

    for item in items_to_evaluate:
        is_method = item['is_method']
        model_name = item['path'].stem if not is_method else item['method']

        do_metrics, do_720p, do_480p = get_columns_to_evaluate(csv_path, model_name)

        if not any([do_metrics, do_720p, do_480p]):
            print(f"Skipping {model_name}")
            continue

        if not is_method:
            checkpoint_path = Path(item['path'])
            log_path = get_logs_path(f"evaluation/eval_{checkpoint_path.stem}.txt")
        else:
            log_path = get_logs_path(f"evaluation/eval_{item['method']}.txt")

        with (Logger(log_path)):
            print("=" * 50)
            print(f"Checkpoint: {checkpoint_path.stem}" if not is_method else f"Method: {item['method']}")
            if USE_HALF: print("Using half precision")

            if is_method:
                model = RegularModel(item['method'], UPSCALE_FACTOR)
            else:
                model, _ = load_model_from_checkpoint(checkpoint_path, device)
                model.upscale_factor = UPSCALE_FACTOR

            # Evaluation
            eval_results = {'LPIPS': "", 'SSIM': "", 'PSNR': "", 'Loss': ""}
            if EVALUATE_METRICS and do_metrics:
                eval_results = evaluator.set_model(model=model).evaluate()

            # Parameter number
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("-" * 30)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print("-" * 30)

            # Performance
            perf_results = {'FPS 720p': "", 'VRAM (MB) 720p': "", 'FPS 480p': "", 'VRAM (MB) 480p': ""}
            if EVALUATE_PERFORMANCE_720p and do_720p and torch.cuda.is_available():
                perf_results['FPS 720p'], perf_results['VRAM (MB) 720p'] = (
                    EvaluatorPerf(model=copy.deepcopy(model), image_size=(720, 1280), use_half=USE_HALF,
                                  use_tensorrt=USE_TENSORRT).evaluate())
            if EVALUATE_PERFORMANCE_480p and do_480p and torch.cuda.is_available():
                perf_results['FPS 480p'], perf_results['VRAM (MB) 480p'] = (
                    EvaluatorPerf(model=copy.deepcopy(model), image_size=(480, 854), use_half=USE_HALF,
                                  use_tensorrt=USE_TENSORRT).evaluate())

            # Save to CSV
            save_to_csv({
                "model_name": checkpoint_path.stem if not is_method else item['method'],
                "params": total_params,
                **eval_results,
                **perf_results,
            }, csv_path)
