"""
Video I/O wrappers: bake the decoder-format -> model -> display-format conversion into the exported
graph, so the runtime (TensorRT / onnxruntime / torch_tensorrt) gets raw decoder frames and returns
frames ready to show, with no per-frame pre/post-processing on the host.

A wrapper is described by a `VideoIO`: the input layout (what the decoder sends) and the output
layout (what the display wants), both meeting at the model's (1,3,H,W) RGB [0,1] tensor.

    input  "rgb" : (3,H,W) uint8 planar RGB     -- NVDEC (RGBP)
    input  "bgr" : (H,W,3) uint8 packed BGR     -- PyAV (bgr24) / cv2.VideoCapture
    output "bgr"     : (H*s,W*s,3) uint8 BGR          -- cv2 display
    output "rgb_f16" : (1,3,H*s,W*s) fp16 RGB [0,255] -- GPU bicubic downscale before display

"rgb_f16" skips the uint8 quantization for players that downscale the SR output on the GPU:
bicubic interpolate can't run on uint8 CUDA tensors, so a uint8 output would just be converted
back to float (and rounded twice).

Combinations in use:
    PyAV / cv2 + cv2 display : bgr -> bgr  (one engine for both CPU decoders)
    NVDEC + cv2 / GL display : rgb -> rgb_f16  (with or without downscale; one engine for both)
"""

from dataclasses import dataclass
from typing import Literal

import torch

Layout = Literal["rgb", "bgr"]
OutputLayout = Literal["bgr", "rgb_f16"]


@dataclass(frozen=True)
class VideoIO:
    """Input/output layout of a wrapped model; `tag` also names its cached export."""
    input: Layout
    output: OutputLayout

    @property
    def tag(self) -> str:
        return f"{self.input}_{self.output}"

    def input_shape(self, h: int, w: int) -> tuple:
        return (3, h, w) if self.input == "rgb" else (h, w, 3)

    def dummy_input(self, h: int, w: int, device="cuda") -> torch.Tensor:
        return torch.randint(0, 256, self.input_shape(h, w), dtype=torch.uint8, device=device)

    def wrap(self, model) -> "VideoWrapper":
        return VideoWrapper(model, self)


class VideoWrapper(torch.nn.Module):
    """uint8 frame in `io.input` layout -> model -> frame in `io.output` layout."""

    def __init__(self, model, io: VideoIO):
        super().__init__()
        self.model = model
        self.io = io

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0  # [0-255] uint8 -> [0-1]
        if self.io.input == "bgr":
            x = x[..., [2, 1, 0]].permute(2, 0, 1)  # (H,W,3) BGR -> (3,H,W) RGB
        out = self.model(x.unsqueeze(0).to(next(self.model.parameters()).dtype))
        out = torch.clamp(out, 0.0, 1.0) * 255.0
        if self.io.output == "rgb_f16":
            return out.to(dtype=torch.float16)  # (1,3,sH,sW) RGB
        out = out.squeeze(0).permute(1, 2, 0)[..., [2, 1, 0]]  # -> (sH,sW,3) BGR
        return out.to(dtype=torch.uint8)
