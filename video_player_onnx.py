from pathlib import Path
import numpy as np
import onnxruntime as ort
from utils.checkpoints import load_model_from_checkpoint
from utils.video.evaluator_perf_video import Runtype, VideoWrapperCV2
from utils.video.export import export_onnx
from utils.video.model_utils import TileProcessor
from utils.video.videoplayer import VideoPlayer

MODEL = "SR_RFDN_2x_2_64"
VIDEO_PATH = Path("videoinput/F1Bahr-480p50.mp4")
UPSCALE_FACTOR = 2
INPUT_SIZE = (480, 854)
TILED = True
TILE_SIZE = 256
RUNTYPE: Runtype = 'onnxruntime-tensorrt'

################################################


if TILED:
    INPUT_SIZE = (TILE_SIZE, TILE_SIZE)

model_path = Path(f"exports/onnx/{MODEL}_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_{UPSCALE_FACTOR}x_cv2.onnx")

# Export if needed
if not model_path.exists():
    print(f"Model export for input size {INPUT_SIZE[0]}x{INPUT_SIZE[1]} doesnt exist")
    print("Exporting...")

    checkpoint_path = Path(f"checkpoints/{MODEL}.pth")
    if not checkpoint_path.exists():
        print(f"Checkpoint for {MODEL} doesnt exit")
        exit()
    model, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
    model.half()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(VideoWrapperCV2(model), model_path, (INPUT_SIZE[0], INPUT_SIZE[1], 3))

# Load model

if RUNTYPE == "onnxruntime-tensorrt":

    providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': True})]
elif RUNTYPE == "onnxruntime-cuda":
    providers = ['CUDAExecutionProvider']
elif RUNTYPE == "onnxruntime-openvino":
    providers = [('OpenVINOExecutionProvider', {"device_type": "GPU", "precision": "FP16"})]

ort_session = ort.InferenceSession(model_path, providers=providers)


# Define callback for model inference
def infer(tile):
    return ort_session.run(None, {"input": tile.astype(np.float16)})[0]


# Define callback for upscaling the frame
if TILED:
    tile_processor = TileProcessor(upscale_factor=UPSCALE_FACTOR, tile_size=TILE_SIZE, overlap=8)


    def upscale(frame):
        return tile_processor.process_frame(frame, infer)
else:
    def upscale(frame):
        return infer(frame)

# Start VideoPlayer
VideoPlayer(video_path=VIDEO_PATH, upscale_fn=upscale).play()

##########################
# ZA INTEL
# mora compatibilnost sa instaliranim onnx paketom da se proveri
# https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#requirements
# onda se skida ARCHIVE
# https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_BASE&VERSION=v_2025_4_1&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE
# i python API za openvino
# pip install openvino==2025.4.1
#
# Pokretanje skripte:
# ...\openvino_toolkit_windows_2025.4.1\setupvars.ps1
# py ./video_player_onnx.py
############################
