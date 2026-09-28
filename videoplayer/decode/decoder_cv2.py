from typing import Optional

import cv2
import numpy as np


class CV2Decoder:
    """
    OpenCV (cv2.VideoCapture) CPU video decoder returning each frame as an (H,W,3) uint8 BGR numpy array

    Note: cv2 ignores the stream's colour tags and always converts with BT.601, so BT.709-tagged
    videos come out with slightly off colours compared to PyAV / NVDEC.

    Decoding is sequential; `frame(index)` for any index other than the next one seeks with
    CAP_PROP_POS_FRAMES first.
    """

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_index = 0
        self._last: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.num_frames

    def frame(self, index: int) -> np.ndarray:
        """Decode frame `index` and return it as an (H,W,3) uint8 BGR array."""
        index = max(0, min(index, self.num_frames - 1))
        if index != self._next_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        ok, frame = self.cap.read()
        if not ok:
            # The reported frame count was too high: shrink it so the reader stops here.
            self.num_frames = index
            self._next_index = index
            if self._last is None:
                raise RuntimeError("CV2Decoder: no video frames could be decoded")
            return self._last

        self._next_index = index + 1
        self._last = frame
        return frame

    def close(self):
        self.cap.release()
