import gc

import torch


def export_onnx_bare(model, path, input_hw):
    """Eksportuje model u fp32 onnx.
    Koristi se za ncnn (pnnx). Mora fp32 jer pnnx ne pravi fp16 slojeve ispravno.
    pnnx svakako sam pretvori iz fp32 u fp16."""
    model = model.eval().cpu().float()
    dummy = torch.rand(1, 3, input_hw[0], input_hw[1])
    torch.onnx.export(model, dummy, str(path), input_names=["input"],
                      output_names=["output"], opset_version=17, dynamo=False)


def export_onnx_uint8(model, onnx_path, input_hw):
    """Eksportuje model u ONNX sa statickim uint8 (H,W,3) ulazom na GPU-u.
    Koristi se za VideoWrapperCV2 wrapovane modele."""
    model = model.eval().cuda().half()
    dummy = torch.randint(0, 256, (input_hw[0], input_hw[1], 3), device="cuda", dtype=torch.uint8)
    try:
        torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"],
                          output_names=["output"], opset_version=17, dynamo=False)
    finally:
        # Oslobodi VRAM koji je export zauzeo
        model.cpu()
        del model, dummy
        gc.collect()
        torch.cuda.empty_cache()


def export_trt(model, path, input_size, use_fp32=False):
    """Eksportuje kompajliran model u .pt2"""
    import torch_tensorrt

    dtype = torch.float32 if use_fp32 else torch.float16
    compiled_model = torch_tensorrt.compile(
        model, inputs=[torch_tensorrt.Input(input_size, dtype=dtype)],
        enabled_precisions={dtype}, offload_module_to_cpu=True)

    torch_tensorrt.save(compiled_model, str(path))

    return compiled_model
