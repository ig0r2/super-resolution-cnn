import sys
from pathlib import Path
from typing import Literal

import torch

# torch.onnx prints unicode (✅) on export; keep it from crashing a cp1252 console/logger.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

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
    "tensorrt", "tensorrt-pt2", "tensorrt-nvdec", "ncnn-vulkan",
    "onnxruntime-cuda", "onnxruntime-tensorrt", "onnxruntime-openvino",
    "onnxruntime-directml", "onnxruntime-cpu"]

NVDEC_RUNTYPE = "tensorrt-nvdec"
STAGES = ("decode", "sr", "e2e", "full")

if __name__ == "__main__":
    UPSCALE_FACTOR: Literal[2, 3, 4] = 2
    RUNTYPE: Runtype = "tensorrt"  # tensorrt | tensorrt-nvdec | onnxruntime-* | ncnn-vulkan

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

            # Load once for param count + config name; the backends reload the checkpoint themselves.
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

                print(f"\n>>> {model_name} @ {label} ({video_path.name}) [{RUNTYPE}]")
                try:
                    metrics = build_evaluator(checkpoint_path, name, video_path).evaluate()
                except RuntimeError as e:
                    print(f"Skipping ({name} @ {label}): {e}")
                    torch.cuda.empty_cache()
                    metrics = {stage: "" for stage in STAGES}

                for stage in STAGES:
                    row[f"{label} {stage}"] = metrics[stage]

                save_to_csv(row, csv_path, match_keys=("model_name", "runtype"))
