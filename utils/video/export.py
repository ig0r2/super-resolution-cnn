import torch


def export_onnx(model, path, input_size, use_fp32=False):
    dtype = torch.float32 if use_fp32 else torch.float16
    input_tensor = torch.randn(input_size).to(dtype).detach()
    torch.onnx.export(model, input_tensor, path, input_names=["input"],
                      output_names=["output"], opset_version=18, external_data=False)


def export_trt(model, path, input_size, use_fp32=False):
    import torch_tensorrt

    dtype = torch.float32 if use_fp32 else torch.float16
    compiled_model = torch_tensorrt.compile(
        model, inputs=[torch_tensorrt.Input(input_size, dtype=dtype)],
        enabled_precisions={dtype}, offload_module_to_cpu=True)

    torch_tensorrt.save(compiled_model, str(path))

    return compiled_model
