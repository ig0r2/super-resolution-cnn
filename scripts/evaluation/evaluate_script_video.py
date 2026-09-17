import sys
from pathlib import Path
from typing import Literal

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.checkpoints import load_model_from_checkpoint
from utils.csv_utils import save_to_csv, already_done
from utils.path import get_logs_path, get_results_path, get_checkpoints_path, get_project_root
from utils.logger import Logger
from videoplayer.evaluator_perf_video_cv2 import EvaluatorPerfVideoCV2
from videoplayer_nvdec.evaluator_perf_video_nvdec import EvaluatorPerfVideoNVDEC

# CPU decode (videoplayer/, cv2.VideoCapture): "tensorrt" / "tensorrt-pt2" / "onnxruntime-*" / "ncnn-vulkan"
# GPU decode (videoplayer_nvdec/, NVDEC):      "tensorrt-nvdec"
# "tensorrt-pt2" is the torch_tensorrt .pt2 path: use it for large models where "tensorrt  engine build OOMs

Runtype = Literal[
    "tensorrt", "tensorrt-pt2", "tensorrt-nvdec", "ncnn-vulkan", "onnxruntime-cuda",
    "onnxruntime-tensorrt", "onnxruntime-openvino", "onnxruntime-directml", "onnxruntime-cpu"]

NVDEC_RUNTYPE = "tensorrt-nvdec"
STAGES = ("decode", "sr", "e2e", "full")


############################################################

if __name__ == "__main__":
    UPSCALE_FACTOR: Literal[2, 3, 4] = 2
    RUNTYPE: Runtype = "tensorrt-nvdec"

    SKIP_EVALUATED = True
    WARMUP_RUNS = 10
    ITERATIONS = 100

    VIDEOS = [
        ("480p", get_project_root("videoinput/ldv-001-480p.mp4")),
    ]

    CHECKPOINT_PATHS = [
        # multiscale - EDSR
        get_checkpoints_path("multiscale/SR_EDSR_2_52.pth"),
        get_checkpoints_path("multiscale/SR_EDSR_4_52.pth"),
        # multiscale - FastEDSR
        get_checkpoints_path("multiscale/SR_FastEDSR_2_4.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_2_8.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_2_16.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_2_32.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_2_64.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_2_128.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_4_8.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_4_16.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_4_32.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_jpeg_4_48.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_4_64.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_4_128.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_4_256.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_jpeg_6_32.pth"),
        get_checkpoints_path("multiscale/SR_FastEDSR_6_256.pth"),
        # multiscale - IMDN
        get_checkpoints_path("multiscale/SR_IMDN_2_48.pth"),
        get_checkpoints_path("multiscale/SR_IMDN_2_96.pth"),
        # multiscale - RFDN
        get_checkpoints_path("multiscale/SR_RFDN_1_128.pth"),
        get_checkpoints_path("multiscale/SR_RFDN_2_48.pth"),
        get_checkpoints_path("multiscale/SR_RFDN_2_96.pth"),
        get_checkpoints_path("multiscale/SR_RFDN_jpeg_2_128.pth"),
        get_checkpoints_path("multiscale/SR_RFDN_jpeg_2_256.pth"),
        get_checkpoints_path("multiscale/SR_RFDN_4_16.pth"),
        get_checkpoints_path("multiscale/SR_RFDN_4_48.pth"),
        get_checkpoints_path("multiscale/SR_RFDN_4_128.pth"),
        get_checkpoints_path("multiscale/SR_RFDN_4_256.pth"),
    ]

    if UPSCALE_FACTOR == 2:
        CHECKPOINT_PATHS += [
            # 2x - EDSR
            get_checkpoints_path("2x/SR_EDSR_2x_0_32.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_1_32.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_1_128.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_1_256.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_2_32.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_2_48.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_2_52.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_2_96.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_2_128.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_2_256.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_4_32.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_4_52.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_4_256.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_6_12.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_16_64.pth"),
            get_checkpoints_path("2x/SR_EDSR_2x_32_256_r.pth"),
            # 2x - FastEDSR
            get_checkpoints_path("2x/SR_FastEDSR_2x_1_4.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_1_8.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_1_16.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_1_32.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_1_64.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_1_128.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_2_4.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_2_8.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_2_16.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_2_32.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_2_64.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_2_128.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_4_8.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_4_16.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_4_32.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_4_64.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_4_128.pth"),
            get_checkpoints_path("2x/SR_FastEDSR_2x_8_16.pth"),
            # 2x - IMDN
            get_checkpoints_path("2x/SR_IMDN_2x_1_48.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_1_64.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_1_96.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_1_128.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_1_256.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_2_48.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_2_96.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_2_128.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_4_16.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_4_52.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_4_256.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_5_16.pth"),
            get_checkpoints_path("2x/SR_IMDN_2x_6_64.pth"),
            # 2x - RFDN
            get_checkpoints_path("2x/SR_RFDN_2x_1_4.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_1_48.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_1_64.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_1_96.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_1_128.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_1_256.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_2_16.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_2_48.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_2_52.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_2_64.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_2_96.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_2_128.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_2_256.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_4_16.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_4_48.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_4_52.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_4_96.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_4_128.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_4_256.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_6_48.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_6_52.pth"),
            get_checkpoints_path("2x/SR_RFDN_2x_6_64.pth"),
            # 2x - SRCNN
            get_checkpoints_path("2x/SR_SRCNN_2x.pth"),
            # 2x - VDSR
            get_checkpoints_path("2x/SR_VDSR_2x_18_64.pth"),
        ]

    if UPSCALE_FACTOR == 3:
        CHECKPOINT_PATHS += [
            # 3x - EDSR
            get_checkpoints_path("3x/SR_EDSR_3x_2_48.pth"),
            get_checkpoints_path("3x/SR_EDSR_3x_2_64.pth"),
            get_checkpoints_path("3x/SR_EDSR_3x_4_48.pth"),
            get_checkpoints_path("3x/SR_EDSR_3x_6_12.pth"),
            # 3x - FastEDSR
            get_checkpoints_path("3x/SR_FastEDSR_3x_1_4.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_1_8.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_1_16.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_1_32.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_1_64.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_1_128.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_2_4.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_2_8.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_2_16.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_2_32.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_2_64.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_2_128.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_4_8.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_4_16.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_4_32.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_4_64.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_4_128.pth"),
            get_checkpoints_path("3x/SR_FastEDSR_3x_8_16.pth"),
            # 3x - IMDN
            get_checkpoints_path("3x/SR_IMDN_3x_2_48.pth"),
            get_checkpoints_path("3x/SR_IMDN_3x_4_48.pth"),
            get_checkpoints_path("3x/SR_IMDN_3x_6_64.pth"),
            # 3x - RFDN
            get_checkpoints_path("3x/SR_RFDN_3x_1_128.pth"),
            get_checkpoints_path("3x/SR_RFDN_3x_2_48.pth"),
            get_checkpoints_path("3x/SR_RFDN_3x_2_128.pth"),
            get_checkpoints_path("3x/SR_RFDN_3x_4_48.pth"),
            get_checkpoints_path("3x/SR_RFDN_3x_6_48.pth"),
            get_checkpoints_path("3x/SR_RFDN_3x_6_52.pth"),
        ]

    if UPSCALE_FACTOR == 4:
        CHECKPOINT_PATHS += [
            # 4x - EDSR
            get_checkpoints_path("4x/SR_EDSR_4x_2_48.pth"),
            get_checkpoints_path("4x/SR_EDSR_4x_2_64.pth"),
            get_checkpoints_path("4x/SR_EDSR_4x_4_48.pth"),
            get_checkpoints_path("4x/SR_EDSR_4x_6_12.pth"),
            # 4x - ESRGAN
            get_checkpoints_path("4x/SR_ESRGAN_4x_23_64_32.pth"),
            # 4x - FastEDSR
            get_checkpoints_path("4x/SR_FastEDSR_4x_1_4.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_1_8.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_1_16.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_1_32.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_1_64.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_1_128.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_2_4.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_2_8.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_2_16.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_2_32.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_2_64.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_2_128.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_4_8.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_4_16.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_4_32.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_4_64.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_4_128.pth"),
            get_checkpoints_path("4x/SR_FastEDSR_4x_8_16.pth"),
            # 4x - IMDN
            get_checkpoints_path("4x/SR_IMDN_4x_2_48.pth"),
            get_checkpoints_path("4x/SR_IMDN_4x_4_48.pth"),
            get_checkpoints_path("4x/SR_IMDN_4x_6_64.pth"),
            # 4x - RFDN
            get_checkpoints_path("4x/SR_RFDN_4x_1_128.pth"),
            get_checkpoints_path("4x/SR_RFDN_4x_2_48.pth"),
            get_checkpoints_path("4x/SR_RFDN_4x_2_128.pth"),
            get_checkpoints_path("4x/SR_RFDN_4x_4_48.pth"),
            get_checkpoints_path("4x/SR_RFDN_4x_6_48.pth"),
            get_checkpoints_path("4x/SR_RFDN_4x_6_52.pth"),
        ]

    #####################################################
    if len(CHECKPOINT_PATHS) == 0:
        CHECKPOINT_PATHS = sorted(p for pattern in (f"{UPSCALE_FACTOR}x/*.pth", "multiscale/*.pth")
                                  for p in get_checkpoints_path().glob(pattern)
                                  if not p.name.endswith("_latest.pth"))

    csv_path = get_results_path(f"results_{UPSCALE_FACTOR}x_video_FPS.csv")

    expected_cols = [f"{label} {stage}" for label, _ in VIDEOS for stage in STAGES]


    def build_evaluator(checkpoint_path, name, video_path):
        if RUNTYPE == NVDEC_RUNTYPE:
            return EvaluatorPerfVideoNVDEC(
                checkpoint_path=checkpoint_path, name=name, video_path=video_path,
                upscale_factor=UPSCALE_FACTOR, warmup_runs=WARMUP_RUNS, iterations=ITERATIONS)
        return EvaluatorPerfVideoCV2(
            checkpoint_path=checkpoint_path, name=name, video_path=video_path,
            runtype=RUNTYPE, upscale_factor=UPSCALE_FACTOR,
            warmup_runs=WARMUP_RUNS, iterations=ITERATIONS)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    for checkpoint_path in CHECKPOINT_PATHS:
        checkpoint_path = Path(checkpoint_path)
        model_name = checkpoint_path.stem

        if SKIP_EVALUATED and already_done(csv_path, model_name, RUNTYPE, expected_cols):
            print(f"Skipping {model_name}")
            continue

        log_path = get_logs_path(f"evaluation/eval_{RUNTYPE}_{checkpoint_path.stem}.txt")

        with Logger(log_path):
            print("=" * 50)
            print(f"Checkpoint: {checkpoint_path.stem}")

            # Load once only for the param count; the backends reload the checkpoint themselves.
            model, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
            model.upscale_factor = UPSCALE_FACTOR
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

                print(f"\n>>> {model_name} @ {label} ({video_path.name}) [{RUNTYPE}]")
                try:
                    metrics = build_evaluator(checkpoint_path, model_name, video_path).evaluate()
                except RuntimeError as e:
                    print(f"Skipping ({model_name} @ {label}): {e}")
                    torch.cuda.empty_cache()
                    metrics = {stage: "" for stage in STAGES}

                for stage in STAGES:
                    row[f"{label} {stage}"] = metrics[stage]

                save_to_csv(row, csv_path, match_keys=("model_name", "runtype"))
