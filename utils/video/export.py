import torch


def export_onnx_bare(model, path, input_hw):
    """Export the bare SR model to ONNX (CHW RGB, fp32) for ncnn (pnnx) conversion.

    Unlike export_onnx_raw (which wraps VideoWrapperCV2: HWC BGR with baked BGR<->RGB / normalize /
    permute), this exports the plain model, so NCNNRunner keeps doing that pre/post itself. Static
    example shape; a fully-conv SR net stays resolution-independent, and the ncnn cache is keyed per
    resolution anyway.
    """
    model = model.eval().cpu().float()
    dummy = torch.rand(1, 3, input_hw[0], input_hw[1])
    torch.onnx.export(model, dummy, str(path), input_names=["input"],
                      output_names=["output"], opset_version=17, dynamo=False)


def export_trt(model, path, input_size, use_fp32=False):
    import torch_tensorrt

    dtype = torch.float32 if use_fp32 else torch.float16
    compiled_model = torch_tensorrt.compile(
        model, inputs=[torch_tensorrt.Input(input_size, dtype=dtype)],
        enabled_precisions={dtype}, offload_module_to_cpu=True)

    torch_tensorrt.save(compiled_model, str(path))

    return compiled_model
