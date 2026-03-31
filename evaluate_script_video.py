import copy
from pathlib import Path
from typing import Literal

import torch

from utils.checkpoints import load_model_from_checkpoint
from utils.csv_utils import save_to_csv
from utils.video.evaluator_perf_video import EvaluatorPerfVideo, Runtype
from utils.logger import Logger

# Evaluacija brzine modela za OpenCV video player demo

if __name__ == "__main__":
    UPSCALE_FACTOR: Literal[2, 3, 4] = 2
    RUNTYPE: Runtype = "tensorrt"

    CHECKPOINT_PATHS = [
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

    #####################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    csv_path = f"results/results_{UPSCALE_FACTOR}x_FPS.csv"

    for checkpoint_path in CHECKPOINT_PATHS:
        checkpoint_path = Path(checkpoint_path)
        log_path = f"logs/evaluation/eval_{checkpoint_path.stem}.txt"

        with (Logger(log_path)):
            print("=" * 50)
            print(f"Checkpoint: {checkpoint_path.stem}")

            model, model_conf = load_model_from_checkpoint(checkpoint_path, device)
            name = model_conf['checkpoint_name']

            # Parameter number
            total_params = sum(p.numel() for p in model.parameters())
            print("-" * 30)
            print(f"Total parameters: {total_params:,}")
            print("-" * 30)

            # Performance
            perf_results = {'720p': "", '480p': "", '720p (128x128)': "", '480p (128x128)': "", '720p (256x256)': "",
                            '480p (256x256)': ""}
            perf_results['720p'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(720, 1280),
                                   upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE).evaluate())
            perf_results['480p'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(480, 854),
                                   upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE).evaluate())
            perf_results['720p (128x128)'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(720, 1280), tiled=True,
                                   tile_size=128, upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE).evaluate())
            perf_results['480p (128x128)'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(480, 854), tiled=True,
                                   tile_size=128, upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE,
                                   dont_export=True).evaluate())
            perf_results['720p (256x256)'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(720, 1280), tiled=True,
                                   tile_size=256, upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE).evaluate())
            perf_results['480p (256x256)'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(480, 854), tiled=True,
                                   tile_size=256, upscale_factor=UPSCALE_FACTOR, runtype=RUNTYPE,
                                   dont_export=True).evaluate())

            # Save to CSV
            save_to_csv({
                "model_name": checkpoint_path.stem,
                "params": total_params,
                "runtype": RUNTYPE,
                **perf_results,
            }, csv_path)
