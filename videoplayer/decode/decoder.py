import torch
import PyNvVideoCodec as nvc


class NvDecoder:
    """
    NVDEC hardware video decoder (PyNvVideoCodec) that yields frames as RGB, channels-first
    (3,H,W) uint8 tensors living on the GPU, ready to feed straight into a TensorRT SR engine.

    Frames are decoded entirely on the GPU (no CPU decode, no host<->device copy on the input
    side) and requested by index, so seeking is just an index jump. `frame()` returns a cloned
    tensor because the decoder recycles a small internal buffer pool across calls.
    """

    def __init__(self, video_path: str, gpu_id: int = 0):
        self.decoder = nvc.SimpleDecoder(
            str(video_path),
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGBP,  # planar RGB -> (3,H,W)
        )
        md = self.decoder.get_stream_metadata()
        self.width = int(md.width)
        self.height = int(md.height)
        self.fps = float(md.average_fps) or 30.0
        self.num_frames = int(md.num_frames)

    def __len__(self) -> int:
        return self.num_frames

    def frame(self, index: int) -> torch.Tensor:
        """Decode frame `index` and return a (3,H,W) uint8 RGB CUDA tensor (a private copy)."""
        index = max(0, min(index, self.num_frames - 1))
        return torch.from_dlpack(self.decoder[index]).clone()
