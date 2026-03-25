import time
from collections import deque
from pathlib import Path
import cv2
import torch
import torch_tensorrt

from utils.video.model_utils import TileProcessorTorch
from utils.video.videostream import VideoStream

MODEL_PATH = Path("exports/SR_Tiny_Res_2x_4_64_480x854_cv2.pt2")
VIDEO_PATH = Path("videoinput/F1Bahr-480p50.mp4")
UPSCALE_FACTOR = 2
TILED = False
TILE_SIZE = 128

################################################

model = torch.export.load(MODEL_PATH).module()


def infer(tile):
    return model(tile)


stream = VideoStream(VIDEO_PATH).start()
h, w = stream.frame_size

tile_processor = TileProcessorTorch(h=h, w=w, c=3, upscale_factor=UPSCALE_FACTOR, tile_size=TILE_SIZE, overlap=8)

frame_times = deque(maxlen=20)  # for FPS avg calculation

while stream.running:
    t0 = time.perf_counter()
    try:
        frame = stream.read()
    except:
        break

    frame_gpu = torch.from_numpy(frame).half().cuda()
    output = tile_processor.process_frame(frame_gpu, infer) if TILED else infer(frame_gpu)
    output = output.cpu().numpy()

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
