import torch


class VideoWrapperNVDEC(torch.nn.Module):
    """
    NVDEC-decode counterpart of utils.video.wrapper.VideoWrapperCV2.

    The NVDEC decoder already hands us an RGB, channels-first (3,H,W) tensor on the GPU, so
    unlike the OpenCV path there is no BGR<->RGB swap and no HWC<->CHW permutation to do here:
    just scale to [0,1], run the model, and scale back to uint8. Values are treated as 0..255
    regardless of dtype (the ONNX graph is traced in fp16), matching VideoWrapperCV2.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x / 255.0).unsqueeze(0)  # (3,H,W) -> (1,3,H,W), [0,1]
        out = self.model(x)
        out = torch.clamp(out, 0.0, 1.0) * 255.0
        out = out.squeeze(0)  # (1,3,H*s,W*s) -> (3,H*s,W*s)
        return out.to(dtype=torch.uint8)


def export_onnx_chw(model, onnx_path, input_hw):
    """Export the model to ONNX (FP16) with a static channels-first (3,H,W) GPU input.

    Mirrors utils.video.export_trt_engine.export_onnx_raw but with a CHW dummy instead of the
    OpenCV (H,W,3) one, so the resulting TensorRT engine consumes NVDEC RGBP frames directly.
    """
    dtype = torch.float16
    wrapper = model.eval().cuda().to(dtype)
    dummy = torch.randn(3, input_hw[0], input_hw[1], device="cuda", dtype=dtype)
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"], opset_version=17, dynamo=False,
    )
