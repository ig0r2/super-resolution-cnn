import copy
from pathlib import Path
from typing import Literal

import torch

from models import RegularModel
from utils.checkpoints import load_model_from_checkpoint
from utils.csv_utils import save_to_csv
from utils.evaluator import Evaluator
from utils.evaluator_perf import EvaluatorPerf
from datasets import get_test_set
from utils.logger import Logger

# Evaluacija modela i/ili metoda

if __name__ == "__main__":
    UPSCALE_FACTOR: Literal[2, 3, 4] = 4
    TEST_SET: Literal["DIV2K", "Set5", "Set14", "BSD100", "Urban100"] = "DIV2K"

    USE_HALF = True
    EVALUATE_METRICS = True
    JPEG_DEGRADATION = True

    EVALUATE_PERFORMANCE_720p = False
    EVALUATE_PERFORMANCE_480p = False
    USE_TENSORRT = True

    CHECKPOINT_PATHS = [
        Path("checkpoints/SR_FastEDSR_jpeg_2_64.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_32.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_32_s.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_32_s.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_64.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_64_s.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_128.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_128_sharp5_1_05.pth"),

        # Path("checkpoints/SR_EDSR_2_52.pth"),
        # Path("checkpoints/SR_EDSR_4_52.pth"),
        # Path("checkpoints/SR_RFDN_2_48.pth"),
        # Path("checkpoints/SR_RFDN_4_48.pth"),
        # Path("checkpoints/SR_RFDN_4_16.pth"),
        # Path("checkpoints/SR_RFDN_2_96.pth"),
        # Path("checkpoints/SR_IMDN_2_48.pth"),
        # Path("checkpoints/SR_IMDN_2_96.pth"),
        # Path("checkpoints/SR_FastEDSR_2_4.pth"),
        # Path("checkpoints/SR_FastEDSR_2_8.pth"),
        # Path("checkpoints/SR_FastEDSR_2_16.pth"),
        # Path("checkpoints/SR_FastEDSR_2_32.pth"),
        # Path("checkpoints/SR_FastEDSR_2_64.pth"),
        # Path("checkpoints/SR_FastEDSR_2_128.pth"),
        # Path("checkpoints/SR_FastEDSR_4_8.pth"),
        # Path("checkpoints/SR_FastEDSR_4_16.pth"),
        # Path("checkpoints/SR_FastEDSR_4_32.pth"),
        # Path("checkpoints/SR_FastEDSR_4_64.pth"),
        # Path("checkpoints/SR_FastEDSR_4_128.pth"),

        # Path("checkpoints/SR_EDSR_2x_1_32.pth"),
        # Path("checkpoints/SR_EDSR_2x_1_128.pth"),
        # Path("checkpoints/SR_EDSR_2x_1_256.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_32.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_48.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_52.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_96.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_128.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_256.pth"),
        # Path("checkpoints/SR_EDSR_2x_4_32.pth"),
        # Path("checkpoints/SR_EDSR_2x_4_52.pth"),
        # Path("checkpoints/SR_EDSR_2x_4_256.pth"),
        # Path("checkpoints/SR_EDSR_2x_6_12.pth"),
        # Path("checkpoints/SR_EDSR_2x_16_64.pth"),
        # Path("checkpoints/SR_EDSR_2x_32_256_r.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_48.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_48.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_52.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_48.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_52.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_16.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_16.pth"),
        # Path("checkpoints/SR_RFDN_2x_6_48.pth"),
        # Path("checkpoints/SR_RFDN_2x_6_52.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_64.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_64.pth"),
        # Path("checkpoints/SR_RFDN_2x_6_64.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_96.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_96.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_96.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_128.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_128.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_128.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_256.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_256.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_256.pth"),
        # Path("checkpoints/SR_IMDN_2x_1_48.pth"),
        # Path("checkpoints/SR_IMDN_2x_2_48.pth"),
        # Path("checkpoints/SR_IMDN_2x_4_16.pth"),
        # Path("checkpoints/SR_IMDN_2x_5_16.pth"),
        # Path("checkpoints/SR_IMDN_2x_4_52.pth"),
        # Path("checkpoints/SR_IMDN_2x_1_64.pth"),
        # Path("checkpoints/SR_IMDN_2x_6_64.pth"),
        # Path("checkpoints/SR_IMDN_2x_1_96.pth"),
        # Path("checkpoints/SR_IMDN_2x_2_96.pth"),
        # Path("checkpoints/SR_IMDN_2x_1_128.pth"),
        # Path("checkpoints/SR_IMDN_2x_2_128.pth"),
        # Path("checkpoints/SR_IMDN_2x_1_256.pth"),
        # Path("checkpoints/SR_IMDN_2x_4_256.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_1_4.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_1_8.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_1_16.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_1_32.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_1_64.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_1_128.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_2_4.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_2_8.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_2_16.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_2_32.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_2_64.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_2_128.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_4_8.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_4_16.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_4_32.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_4_64.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_4_128.pth"),
        # Path("checkpoints/SR_FastEDSR_2x_8_16.pth"),

        # Path("checkpoints/SR_EDSR_3x_6_12.pth"),
        # Path("checkpoints/SR_EDSR_3x_2_48.pth"),
        # Path("checkpoints/SR_EDSR_3x_4_48.pth"),
        # Path("checkpoints/SR_EDSR_3x_2_64.pth"),
        # Path("checkpoints/SR_RFDN_3x_2_48.pth"),
        # Path("checkpoints/SR_RFDN_3x_4_48.pth"),
        # Path("checkpoints/SR_RFDN_3x_6_48.pth"),
        # Path("checkpoints/SR_RFDN_3x_6_52.pth"),
        # Path("checkpoints/SR_RFDN_3x_1_128.pth"),
        # Path("checkpoints/SR_RFDN_3x_2_128.pth"),
        # Path("checkpoints/SR_IMDN_3x_2_48.pth"),
        # Path("checkpoints/SR_IMDN_3x_4_48.pth"),
        # Path("checkpoints/SR_IMDN_3x_6_64.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_1_4.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_1_8.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_1_16.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_1_32.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_1_64.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_1_128.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_2_4.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_2_8.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_2_16.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_2_32.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_2_64.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_2_128.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_4_8.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_4_16.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_4_32.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_4_64.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_4_128.pth"),
        # Path("checkpoints/SR_FastEDSR_3x_8_16.pth"),

        # Path("checkpoints/SR_EDSR_4x_6_12.pth"),
        # Path("checkpoints/SR_EDSR_4x_2_48.pth"),
        # Path("checkpoints/SR_EDSR_4x_4_48.pth"),
        # Path("checkpoints/SR_EDSR_4x_2_64.pth"),
        # Path("checkpoints/SR_RFDN_4x_2_48.pth"),
        # Path("checkpoints/SR_RFDN_4x_4_48.pth"),
        # Path("checkpoints/SR_RFDN_4x_6_48.pth"),
        # Path("checkpoints/SR_RFDN_4x_6_52.pth"),
        # Path("checkpoints/SR_RFDN_4x_1_128.pth"),
        # Path("checkpoints/SR_RFDN_4x_2_128.pth"),
        # Path("checkpoints/SR_IMDN_4x_2_48.pth"),
        # Path("checkpoints/SR_IMDN_4x_4_48.pth"),
        # Path("checkpoints/SR_IMDN_4x_6_64.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_1_4.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_1_8.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_1_16.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_1_32.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_1_64.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_1_128.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_2_4.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_2_8.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_2_16.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_2_32.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_2_64.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_2_128.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_4_8.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_4_16.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_4_32.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_4_64.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_4_128.pth"),
        # Path("checkpoints/SR_FastEDSR_4x_8_16.pth"),
    ]
    METHODS = [
        # 'nearest',
        # 'bilinear',
        # 'bicubic',
        # 'lanczos'
    ]

    #####################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if EVALUATE_METRICS:
        test_set = get_test_set(name="DIV2K", upscale_factor=UPSCALE_FACTOR, preload=len(CHECKPOINT_PATHS) > 1,
                                normalize=False, jpeg_degradation=JPEG_DEGRADATION)
        evaluator = Evaluator(test_set=test_set, device=device, use_half=USE_HALF)

    # Combine checkpoints and methods
    items_to_evaluate = [{'path': p, 'is_method': False} for p in CHECKPOINT_PATHS]
    items_to_evaluate += [{'method': m, 'is_method': True} for m in METHODS]

    csv_path = f"results/results_{UPSCALE_FACTOR}x_{TEST_SET}{'_jpeg' if JPEG_DEGRADATION else ''}{'_half' if USE_HALF else ''}.csv"

    for item in items_to_evaluate:
        is_method = item['is_method']

        if not is_method:
            checkpoint_path = Path(item['path'])
            log_path = f"logs/evaluation/eval_{checkpoint_path.stem}.txt"
        else:
            log_path = f"logs/evaluation/eval_{item['method']}.txt"

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
            if EVALUATE_METRICS:
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
            if EVALUATE_PERFORMANCE_720p and torch.cuda.is_available():
                perf_results['FPS 720p'], perf_results['VRAM (MB) 720p'] = (
                    EvaluatorPerf(model=copy.deepcopy(model), image_size=(720, 1280), use_half=USE_HALF,
                                  use_tensorrt=USE_TENSORRT).evaluate())
            if EVALUATE_PERFORMANCE_480p and torch.cuda.is_available():
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
