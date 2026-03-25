import cv2
import time
import threading
from queue import Queue


# Ucitava video i smesta frejmove u queue velicine 1 i tako simulira realtime video streaming
class VideoStream:
    def __init__(self, src, queue_size=1):
        self.cap = cv2.VideoCapture(src)
        self.frame_time = 1.0 / self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.queue = Queue(maxsize=queue_size)
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        next_time = time.perf_counter()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            # drop frame if queue full
            if not self.queue.full():
                self.queue.put(frame)

            next_time += self.frame_time
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        return self.queue.get(timeout=1)

    def stop(self):
        self.running = False
        self.cap.release()
