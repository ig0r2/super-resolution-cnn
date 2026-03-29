import time
from collections import deque

import cv2

from .videostream import VideoStream


class VideoPlayer:
    def __init__(self, video_path, upscale_fn):
        self.stream = VideoStream(video_path)
        self.upscale_fn = upscale_fn

    def play(self):
        self.stream.start()

        frame_times = deque(maxlen=20)  # for FPS avg calculation

        while self.stream.running:
            t0 = time.perf_counter()
            try:
                frame = self.stream.read()
            except:
                break

            output = self.upscale_fn(frame)

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
