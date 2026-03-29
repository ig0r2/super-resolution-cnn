import torch


def export_onnx(model, path, input_size):
    input_tensor = torch.randn(input_size).half().detach()
    torch.onnx.export(model, input_tensor, path, input_names=["input"],
                      output_names=["output"], opset_version=18, external_data=False)


def export_trt(model, path, input_size):
    import torch_tensorrt

    compiled_model = torch_tensorrt.compile(
        model, inputs=[torch_tensorrt.Input(input_size, dtype=torch.float16)],
        enabled_precisions={torch.float16})

    torch_tensorrt.save(compiled_model, str(path))

    return compiled_model
