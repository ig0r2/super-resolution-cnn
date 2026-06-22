import torch
import torch_tensorrt

from utils.checkpoints import load_model_from_checkpoint
from utils.path import get_project_root, get_checkpoints_path
from utils.video.evaluator_perf_video import VideoWrapperCV2
from utils.video.export import export_trt
from utils.video.model_utils import TileProcessorTorch
from utils.video.videoplayer import VideoPlayer

MODEL = "multiscale/SR_FastEDSR_jpeg_4_32"
VIDEO_PATH = get_project_root("videoinput/F1Bahr-480p50.mp4")
UPSCALE_FACTOR = 2
INPUT_SIZE = (480, 854)
TILED = False
TILE_SIZE = 256

################################################

if TILED:
    INPUT_SIZE = (TILE_SIZE, TILE_SIZE)

checkpoint_name = MODEL.split("/", 1)[-1]

model_path = get_project_root(
    f"exports/trt/{checkpoint_name}_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_{UPSCALE_FACTOR}x_cv2.pt2")

# Export if needed
if not model_path.exists():
    print(f"Model export for input size {INPUT_SIZE[0]}x{INPUT_SIZE[1]} doesnt exist")
    print("Compiling and Exporting...")

    checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")
    if not checkpoint_path.exists():
        print(f"Checkpoint for {MODEL} doesnt exit")
        exit()
    model, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
    model.half()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    export_trt(VideoWrapperCV2(model), model_path, (INPUT_SIZE[0], INPUT_SIZE[1], 3))

# Load model
model = torch.export.load(model_path).module().cuda()


# Define callback for model inference
def infer(tile):
    return model(tile)


# Define callback for upscaling the frame
if TILED:
    tile_processor = TileProcessorTorch(upscale_factor=UPSCALE_FACTOR, tile_size=TILE_SIZE, overlap=8)


    def upscale(frame):
        frame_gpu = torch.from_numpy(frame).half().cuda()
        return tile_processor.process_frame(frame_gpu, infer).cpu().numpy()

else:
    def upscale(frame):
        frame_gpu = torch.from_numpy(frame).half().cuda()
        return infer(frame_gpu).cpu().numpy()

# Start VideoPlayer
VideoPlayer(video_path=VIDEO_PATH, upscale_fn=upscale).play()
