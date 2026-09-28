from typing import Optional

import av
import numpy as np
from av.video.reformatter import Interpolation, VideoReformatter

# swscale's default is nearest-neighbour chroma upsampling with fast rounding; these flags give the
# exact YUV->BGR conversion, at roughly 2x the (sub-millisecond) CPU cost.
_ACCURATE = Interpolation.BILINEAR | Interpolation.ACCURATE_RND | Interpolation.FULL_CHR_H_INT


class PyAVDecoder:
    """
    FFmpeg (PyAV) CPU video decoder with the NvDecoder interface (`len()`, `fps`, `width`/`height`,
    `frame(index)`), returning each frame as an (H,W,3) uint8 BGR numpy array - the same "bgr"
    layout as CV2Decoder, so both CPU decoders share one wrapper/engine.

    YUV->BGR is done on the CPU by swscale, which reads the stream's colour tags (BT.601/709,
    limited/full range; untagged streams are treated as BT.601) -- unlike cv2, which always uses
    BT.601. swscale's packed bgr24 output is already HWC, so no extra transpose/copy is needed.

    FFmpeg decodes sequentially, so `frame(index)` is cheap only when `index` is the next frame
    (the reader's normal case); any other index is a seek to the preceding keyframe followed by
    decoding forward (without conversion) until the requested frame.
    """

    def __init__(self, video_path: str):
        self.container = av.open(str(video_path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"  # frame + slice threading in libavcodec

        cc = self.stream.codec_context
        self.width = int(cc.width)
        self.height = int(cc.height)
        rate = self.stream.average_rate or self.stream.guessed_rate
        self.fps = float(rate) if rate else 30.0

        self._tb = float(self.stream.time_base)
        self._start_pts = self.stream.start_time or 0
        self.num_frames = int(self.stream.frames or 0)
        if self.num_frames <= 0:  # e.g. webm/mkv don't store a frame count -> estimate from duration
            if self.stream.duration is not None:
                duration_s = self.stream.duration * self._tb
            else:
                duration_s = (self.container.duration or 0) / av.time_base
            self.num_frames = max(1, int(round(duration_s * self.fps)))

        # One reformatter for the whole stream: frame.to_ndarray(format=...) builds a fresh swscale
        # context per frame, a fixed ~2.3 ms cost that dominated decode time even at 240p.
        self._reformatter = VideoReformatter()

        self._frames = self.container.decode(self.stream)
        self._next_index = 0
        self._pending: Optional[av.VideoFrame] = None  # frame found by a seek, not yet returned
        self._last: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.num_frames

    def _index_of(self, frame: av.VideoFrame, fallback: int) -> int:
        if frame.pts is None:
            return fallback
        return int(round((frame.pts - self._start_pts) * self._tb * self.fps))

    def _seek(self, index: int):
        target_pts = self._start_pts + int(index / self.fps / self._tb)
        self.container.seek(target_pts, stream=self.stream, backward=True)
        self._frames = self.container.decode(self.stream)
        self._pending = None
        last = None
        for f in self._frames:
            last = f
            if self._index_of(f, index) >= index:
                self._pending = f
                return
        if last is not None:
            # Hit EOF before `index`: the estimated frame count was too high, so the real last
            # frame becomes the end of the video.
            self.num_frames = self._index_of(last, index) + 1
            self._pending = last

    def _convert(self, frame: av.VideoFrame) -> np.ndarray:
        return self._reformatter.reformat(frame, format="bgr24", interpolation=_ACCURATE).to_ndarray()

    def frame(self, index: int) -> np.ndarray:
        """Decode frame `index` and return it as an (H,W,3) uint8 BGR array."""
        index = max(0, min(index, self.num_frames - 1))
        if index != self._next_index:
            self._seek(index)

        f, self._pending = self._pending, None
        if f is None:
            f = next(self._frames, None)
        if f is None:
            # The frame count was an overestimate: shrink it so the reader stops here.
            self.num_frames = index
            self._next_index = index
            if self._last is None:
                raise RuntimeError("PyAVDecoder: no video frames could be decoded")
            return self._last

        self._next_index = index + 1
        self._last = self._convert(f)
        return self._last

    def close(self):
        self.container.close()
