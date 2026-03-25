import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from utils.video.model_utils import TileProcessor
from utils.video.videostream import VideoStream
from utils.video.evaluator_perf_video import Runtype

MODEL_PATH = Path("exports/SR_Tiny_Res_2x_4_64_480x854_cv2.onnx")
VIDEO_PATH = Path("videoinput/F1Bahr-480p50.mp4")
UPSCALE_FACTOR = 2
TILED = False
TILE_SIZE = 128
RUNTYPE: Runtype = 'onnxruntime-tensorrt'

################################################

if RUNTYPE == "onnxruntime-tensorrt":
    import torch_tensorrt

    providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': True})]
elif RUNTYPE == "onnxruntime-cuda":
    providers = ['CUDAExecutionProvider']
elif RUNTYPE == "onnxruntime-openvino":
    providers = [('OpenVINOExecutionProvider', {"device_type": "GPU", "precision": "FP16"})]

ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)


def infer(tile):
    return ort_session.run(None, {"input": tile.astype(np.float16)})[0]


stream = VideoStream(VIDEO_PATH).start()
h, w = stream.frame_size

tile_processor = TileProcessor(h=h, w=w, c=3, upscale_factor=UPSCALE_FACTOR, tile_size=TILE_SIZE, overlap=8)

frame_times = deque(maxlen=20)  # for FPS avg calculation

while stream.running:
    t0 = time.perf_counter()
    try:
        frame = stream.read()
    except:
        break

    output = tile_processor.process_frame(frame, infer) if TILED else infer(frame)

    frame_time_ms = (time.perf_counter() - t0) * 1000
    frame_times.append(frame_time_ms)
    avg_fps = 1000.0 / (sum(frame_times) / len(frame_times))

    print(f"{frame_time_ms:.2f} ms | {avg_fps:.1f} FPS")

    cv2.putText(output, f"{avg_fps:.1f} FPS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("SR Video", output)
    cv2.imshow("Original Video", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

##################
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
####################
