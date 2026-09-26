import gc
import time
from pathlib import Path

import torch


def _log(msg: str):
    print(f"[videoplayer] {msg}")


def load_model(checkpoint_path, upscale_factor):
    """Ucitava model iz checkpoint-a na CPU i postavlja mu upscale faktor."""
    from utils.checkpoints import load_model_from_checkpoint

    _log(f"Loading checkpoint {Path(checkpoint_path).name}")
    t0 = time.perf_counter()
    model, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
    model.upscale_factor = upscale_factor
    _log(f"Checkpoint loaded in {time.perf_counter() - t0:.1f}s")
    return model


def get_onnx_video(checkpoint_path, onnx_path, input_size, upscale_factor, io):
    """Return `onnx_path`, exporting the `io` VideoWrapper ONNX from the checkpoint only if the ONNX
    is missing. A cached ONNX means the .pth is never loaded."""
    if onnx_path.exists():
        _log(f"Reusing cached ONNX {onnx_path.name}")
        return onnx_path

    if not Path(checkpoint_path).exists():
        raise RuntimeError(f"No cached ONNX ({onnx_path.name}) and no checkpoint ({Path(checkpoint_path).name})")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint_path, upscale_factor)
    _log(f"Exporting {io.tag} ONNX ({input_size[0]}x{input_size[1]}) -> {onnx_path.name} ...")
    t0 = time.perf_counter()
    export_onnx_video(io.wrap(model), onnx_path, (input_size[0], input_size[1]))
    _log(f"ONNX export done in {time.perf_counter() - t0:.1f}s")
    return onnx_path


def export_onnx_video(wrapper, onnx_path, input_hw):
    """Eksportuje VideoWrapper u ONNX (FP16 model) sa statickim uint8 ulazom na GPU-u, ciji je
    oblik odredjen wrapper.io ((3,H,W) za rgb, (H,W,3) za bgr)."""
    wrapper = wrapper.eval().cuda().half()
    dummy = wrapper.io.dummy_input(input_hw[0], input_hw[1])
    try:
        torch.onnx.export(wrapper, dummy, str(onnx_path), input_names=["input"],
                          output_names=["output"], opset_version=17, dynamo=False)
    finally:
        # Oslobodi VRAM koji je export zauzeo
        wrapper.cpu()
        del wrapper, dummy
        gc.collect()
        torch.cuda.empty_cache()


def export_trt(wrapper, path, input_hw, use_fp32=False):
    """Eksportuje kompajliran VideoWrapper u .pt2 (fp16 ulaz oblika wrapper.io)"""
    import torch_tensorrt

    dtype = torch.float32 if use_fp32 else torch.float16
    compiled_model = torch_tensorrt.compile(
        wrapper, inputs=[torch_tensorrt.Input(wrapper.io.input_shape(*input_hw), dtype=dtype)],
        enabled_precisions={dtype}, offload_module_to_cpu=True)

    torch_tensorrt.save(compiled_model, str(path))

    return compiled_model
