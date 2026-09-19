import gc
from pathlib import Path

import torch


def load_raw_trt_engine(engine_path):
    """Ucitava vec izgradjen TRT engine iz kesa na disku."""
    import tensorrt as trt
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return runtime.deserialize_cuda_engine(engine_path.read_bytes())


def build_raw_trt_engine(onnx_path, engine_path, workspace_gb=None, opt_level=3):
    """Gradi staticni TRT engine (uvek FP16) preko cistog TensorRT Python API-ja i kesira ga na disk.

    workspace_gb=None auto-sizes the workspace pool to the currently free VRAM (minus
    workspace_headroom_gb), so tactics aren't skipped just because a fixed cap was too tight.
    """
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    if workspace_gb is None:
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        workspace_headroom_gb = 1.0
        workspace_bytes = max(int(free_bytes - workspace_headroom_gb * (1 << 30)), 0)
    else:
        workspace_bytes = int(workspace_gb * (1 << 30))

    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.builder_optimization_level = opt_level
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


def get_raw_trt_engine(onnx_path, engine_path, workspace_gb=None, opt_level=3, rebuild=False):
    """Ucitava keširani TRT engine ako postoji na disku, inace ga gradi (FP16)"""
    if engine_path.exists() and not rebuild:
        return load_raw_trt_engine(engine_path)
    return build_raw_trt_engine(onnx_path, engine_path, workspace_gb=workspace_gb, opt_level=opt_level)


class TRTRawRunner:
    """Pokrece staticni TRT engine (cist TensorRT API, bez torch_tensorrt/.pt2) nad tenzorom (H,W,3)."""

    def __init__(self, engine):
        import tensorrt as trt
        trt_to_torch = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                        trt.DataType.INT8: torch.int8, trt.DataType.INT32: torch.int32, trt.DataType.UINT8: torch.uint8}
        self.engine = engine
        self.context = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_dtype = trt_to_torch[engine.get_tensor_dtype(self.in_name)]
        self.out_dtype = trt_to_torch[engine.get_tensor_dtype(self.out_name)]
        self.out_shape = tuple(engine.get_tensor_shape(self.out_name))
        self.stream = torch.cuda.Stream()

    def __call__(self, x):
        # make sure x is on device
        self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            x = x.to(self.in_dtype).contiguous()
            out = torch.empty(self.out_shape, dtype=self.out_dtype, device="cuda")
            self.context.set_tensor_address(self.in_name, x.data_ptr())
            self.context.set_tensor_address(self.out_name, out.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)

        self.stream.synchronize()
        return out
