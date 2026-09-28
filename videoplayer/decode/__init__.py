"""
Video decoders, all with the same `len()` / `fps` / `height` / `width` / `frame(index)` / `close()`
interface: NVDEC gives (3,H,W) uint8 RGB CUDA tensors, PyAV and cv2 (H,W,3) uint8 BGR numpy arrays.
"""
from typing import Literal

CpuDecoder = Literal["pyav", "cv2"]
Decoder = Literal["nvdec", "pyav", "cv2"]


def open_decoder(video_path, decoder: Decoder):
    if decoder == "nvdec":
        from .decoder_nvdec import NvDecoder
        return NvDecoder(str(video_path))
    if decoder == "pyav":
        from .decoder_pyav import PyAVDecoder
        return PyAVDecoder(str(video_path))
    if decoder == "cv2":
        from .decoder_cv2 import CV2Decoder
        return CV2Decoder(str(video_path))
    raise ValueError(f"Unknown decoder {decoder!r} (expected 'nvdec', 'pyav' or 'cv2')")
