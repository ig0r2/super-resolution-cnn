import gc
from pathlib import Path

import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.UINT8: torch.uint8,
}


class TRTBuildTooLarge(RuntimeError):
    """Raised when the estimated TensorRT workspace need exceeds the VRAM available for a build."""


def estimate_conv_workspace_bytes(model, input_hw, use_fp32=False, headroom_gb=1.0, label=None):
    """Rough upper-bound estimate of the largest per-layer TensorRT workspace this network could
    request at the given resolution, based on the classic im2col/GEMM buffer size
    (in_channels/groups * kernel_h * kernel_w * out_h * out_w * dtype_bytes) for each conv layer.
    """
    dtype = torch.float32 if use_fp32 else torch.float16
    dtype_bytes = 4 if use_fp32 else 2

    peak_bytes = 0

    def hook(module, _inputs, output):
        nonlocal peak_bytes
        out_h, out_w = output.shape[-2], output.shape[-1]
        kh, kw = module.kernel_size
        in_ch = module.in_channels // module.groups
        peak_bytes = max(peak_bytes, in_ch * kh * kw * out_h * out_w * dtype_bytes)

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]

    try:
        wrapper = model.eval().cuda().to(dtype)
        dummy = torch.randn(input_hw[0], input_hw[1], 3, device="cuda", dtype=dtype)
        with torch.no_grad():
            wrapper(dummy)
    finally:
        for h in handles:
            h.remove()

    gc.collect()
    torch.cuda.empty_cache()
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    headroom_bytes = int(headroom_gb * (1 << 30))
    if peak_bytes > max(free_bytes - headroom_bytes, 0):
        prefix = f"{label} " if label else ""
        raise TRTBuildTooLarge(
            f"{prefix}at {input_hw} needs an estimated {peak_bytes / (1 << 30):.2f} GB TensorRT "
            f"workspace, but only {free_bytes / (1 << 30):.2f} GB is free "
            f"({headroom_bytes / (1 << 30):.1f} GB headroom reserved). Skipping build.")

    return peak_bytes


def export_onnx_raw(model, onnx_path, input_hw, use_fp32=False):
    """Eksportuje model u ONNX sa statickim ulazom (H,W,3) na GPU-u, za raw TensorRT build."""
    dtype = torch.float32 if use_fp32 else torch.float16
    wrapper = model.eval().cuda().to(dtype)
    dummy = torch.randn(input_hw[0], input_hw[1], 3, device="cuda", dtype=dtype)
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"], opset_version=17, dynamo=False,
    )


def load_raw_trt_engine(engine_path):
    """Ucitava vec izgradjen TRT engine iz kesa na disku."""
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_path.read_bytes())


def build_raw_trt_engine(onnx_path, engine_path, use_fp32=False, workspace_gb=None, opt_level=3):
    """Gradi staticni TRT engine preko cistog TensorRT Python API-ja i kesira ga na disk.

    workspace_gb=None auto-sizes the workspace pool to the currently free VRAM (minus
    workspace_headroom_gb), so tactics aren't skipped just because a fixed cap was too tight.
    """
    runtime = trt.Runtime(TRT_LOGGER)

    if workspace_gb is None:
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        workspace_headroom_gb = 1.0
        workspace_bytes = max(int(free_bytes - workspace_headroom_gb * (1 << 30)), 0)
    else:
        workspace_bytes = int(workspace_gb * (1 << 30))

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.builder_optimization_level = opt_level
    if not use_fp32:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed (build_serialized_network vratio None)")

    engine_path.write_bytes(bytes(serialized))
    engine = runtime.deserialize_cuda_engine(serialized)

    del builder, network, parser, config, serialized
    gc.collect()
    torch.cuda.empty_cache()
    return engine


def get_raw_trt_engine(onnx_path, engine_path, use_fp32=False, workspace_gb=None, opt_level=3, rebuild=False):
    """Ucitava keširani TRT engine ako postoji na disku, inace ga gradi.

    Tanka dispecerska funkcija oko load_raw_trt_engine / build_raw_trt_engine.
    """
    if engine_path.exists() and not rebuild:
        return load_raw_trt_engine(engine_path)
    return build_raw_trt_engine(onnx_path, engine_path, use_fp32=use_fp32, workspace_gb=workspace_gb,
                                 opt_level=opt_level)


class TRTRawRunner:
    """Pokrece staticni TRT engine (cist TensorRT API, bez torch_tensorrt/.pt2) nad tenzorom (H,W,3)."""

    def __init__(self, engine):
        self.engine = engine
        self.context = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_dtype = _TRT_TO_TORCH[engine.get_tensor_dtype(self.in_name)]
        self.out_dtype = _TRT_TO_TORCH[engine.get_tensor_dtype(self.out_name)]
        self.stream = torch.cuda.Stream()

    def __call__(self, x):
        # make sure x is on device
        self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            x = x.to(self.in_dtype).contiguous()
            self.context.set_input_shape(self.in_name, tuple(x.shape))
            out_shape = tuple(self.context.get_tensor_shape(self.out_name))
            out = torch.empty(out_shape, dtype=self.out_dtype, device="cuda")
            self.context.set_tensor_address(self.in_name, x.data_ptr())
            self.context.set_tensor_address(self.out_name, out.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
            if out.dtype != torch.uint8:  # ako TRT ne izlazi uint8, konvertuj ovde
                out = out.clamp(0, 255).to(torch.uint8)

        self.stream.synchronize()
        return out
