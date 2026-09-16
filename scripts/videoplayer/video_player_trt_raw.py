import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from utils.checkpoints import load_model_from_checkpoint
from utils.path import get_project_root, get_checkpoints_path
from utils.video.evaluator_perf_video import VideoWrapperCV2
from utils.video.export_trt_engine import export_onnx_raw, get_raw_trt_engine, TRTRawRunner
from utils.video.model_utils import TileProcessorTorch
from utils.video.videoplayer import VideoPlayer

MODEL = "multiscale/SR_FastEDSR_jpeg_4_256"
VIDEO_PATH = get_project_root("videoinput/F1Bahr-480p50.mp4")
UPSCALE_FACTOR = 2
TILED = False
TILE_SIZE = 256

################################################

# create player
player = VideoPlayer(VIDEO_PATH)

# get frame size for model input
INPUT_SIZE = (TILE_SIZE, TILE_SIZE) if TILED else player.size

checkpoint_name = MODEL.split("/", 1)[-1]
base_tag = f"{checkpoint_name}_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_{UPSCALE_FACTOR}x_cv2"
onnx_path = get_project_root(f"exports/trt_raw/{base_tag}.onnx")
engine_path = get_project_root(f"exports/trt_raw/{base_tag}.engine")

# Export if needed
if not engine_path.exists():
    print(f"Model export for input size {INPUT_SIZE[0]}x{INPUT_SIZE[1]} doesnt exist")
    print("Compiling and Exporting...")

    checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")
    if not checkpoint_path.exists():
        print(f"Checkpoint for {MODEL} doesnt exit")
        exit()
    model, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
    model.upscale_factor = UPSCALE_FACTOR

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx_raw(VideoWrapperCV2(model), onnx_path, (INPUT_SIZE[0], INPUT_SIZE[1]))

# Load (or build) engine
engine = get_raw_trt_engine(onnx_path, engine_path)
runner = TRTRawRunner(engine)


# Define callback for model inference
def infer(tile):
    return runner(tile)


# Define callback for upscaling the frame
if TILED:
    tile_processor = TileProcessorTorch(upscale_factor=UPSCALE_FACTOR, tile_size=TILE_SIZE, overlap=8)


    def upscale(frame):
        frame_gpu = torch.from_numpy(frame).cuda()
        return tile_processor.process_frame(frame_gpu, infer).cpu().numpy()

else:
    def upscale(frame):
        frame_gpu = torch.from_numpy(frame).cuda()
        return infer(frame_gpu).cpu().numpy()

# Start VideoPlayer
player.set_upscale_fn(upscale).play()
