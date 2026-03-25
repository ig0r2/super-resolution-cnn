from pathlib import Path
from typing import Literal, TypeAlias
import numpy as np
import torch
import time

from utils.video.model_utils import TileProcessor, TileProcessorTorch

Runtype: TypeAlias = Literal['tensorrt', 'onnxruntime-cuda', 'onnxruntime-tensorrt', 'onnxruntime-openvino']


# Pretvara iz OpenCV formata u format za model, odradi inference i onda vrati u format za OpenCV
class VideoWrapperCV2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., [2, 1, 0]] / 255.0  # BGR [0-255] -> RGB [0-1]
        x = x.permute(2, 0, 1).unsqueeze(0)  # (H,W,C) -> (1,C,H,W)
        out = self.model(x)
        out = torch.clamp(out, 0.0, 1.0) * 255.0
        # (1,C,H,W) -> (H,W,C) and RGB -> BGR
        out = out.squeeze(0).permute(1, 2, 0)[..., [2, 1, 0]]
        return out.to(dtype=torch.uint8)


class EvaluatorPerfVideo:
    """
    Evaluacija brzine inferense. Kroz model se pusta random tenzor odredjen broj iteracija i racuna kao FPS
    """

    def __init__(self, model, name, runtype: Runtype, image_size=(720, 1280), tiled=False, tile_size=256,
                 dont_export=False, warmup_runs=10, iterations=100):
        self.model = model
        self.model.eval()
        self.model.half()
        self.name = name

        self.tiled = tiled
        self.tile_size = tile_size
        self.image_size = (image_size[0], image_size[1], 3)
        self.input_frame = np.random.randint(0, 255, self.image_size)
        self.input_size = (tile_size, tile_size, 3) if tiled else self.image_size
        self.input_tensor = torch.randn(self.input_size).half().detach()

        self.warmup_runs = warmup_runs
        self.iterations = iterations
        self.runtype: Runtype = runtype

        self.dont_export = dont_export

    def evaluate(self):
        if self.runtype == 'tensorrt':
            return self.evaluate_tensorrt()
        elif self.runtype.startswith('onnxruntime'):
            return self.evaluate_onnx()

    # ===================ONNX================

    def evaluate_onnx(self):
        import onnxruntime as ort

        # Export model
        output_path = Path(f"exports/{self.name}_{self.input_size[0]}x{self.input_size[1]}_cv2.onnx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.dont_export:
            torch.onnx.export(VideoWrapperCV2(self.model), self.input_tensor, output_path, input_names=["input"],
                              output_names=["output"], opset_version=18, external_data=False)

        if self.runtype == "onnxruntime-tensorrt":
            import torch_tensorrt
            providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': True})]
        elif self.runtype == "onnxruntime-cuda":
            providers = ['CUDAExecutionProvider']
        elif self.runtype == "onnxruntime-openvino":
            providers = [('OpenVINOExecutionProvider', {"device_type": "GPU", "precision": "FP16"})]

        ort_session = ort.InferenceSession(output_path, providers=providers)
        tile_processor = TileProcessor(h=self.input_size[1], w=self.input_size[0], c=3,
                                       upscale_factor=self.model.upscale_factor, tile_size=self.tile_size, overlap=8)

        def onnx_run(tile):
            return ort_session.run(None, {"input": tile.astype(np.float16)})[0]

        # Inference
        print(f"Using input shape: {self.input_size}")

        with torch.no_grad():
            # Warmup
            for _ in range(self.warmup_runs):
                if self.tiled:
                    _ = tile_processor.process_frame(self.input_frame, onnx_run)
                else:
                    _ = onnx_run(self.input_frame)

            # Timing
            start_time = time.time()
            for _ in range(self.iterations):
                if self.tiled:
                    _ = tile_processor.process_frame(self.input_frame, onnx_run)
                else:
                    _ = onnx_run(self.input_frame)

            end_time = time.time()

        total_time = end_time - start_time
        avg_time_ms = (total_time / self.iterations) * 1000

        print("-" * 30)
        print(f"Total time for {self.iterations} runs: {total_time:.4f}s")
        print(f"Average Inference Time: {avg_time_ms:.2f} ms")
        print(f"Throughput: {1.0 / (avg_time_ms / 1000):.2f} FPS")
        print("-" * 30)

        return f"{1.0 / (avg_time_ms / 1000):.2f}"

    # ============TENSORRT=============

    def evaluate_tensorrt(self):
        import torch_tensorrt

        torch.cuda.empty_cache()

        # Compile model
        compiled_model = torch_tensorrt.compile(
            VideoWrapperCV2(self.model),
            inputs=[torch_tensorrt.Input(self.input_size, dtype=torch.float16)],
            enabled_precisions={torch.float16})

        output_path = Path(f"exports/{self.name}_{self.input_size[0]}x{self.input_size[1]}_cv2.pt2")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.dont_export:
            torch_tensorrt.save(compiled_model, str(output_path))

        print(f"Using input shape: {self.input_size}")
        # Inference
        tile_processor = TileProcessorTorch(
            h=self.input_size[1], w=self.input_size[0], c=3,
            upscale_factor=self.model.upscale_factor, tile_size=self.tile_size, overlap=8)

        def infer(tile):
            return compiled_model(tile)

        with torch.no_grad():
            # Warmup
            for _ in range(self.warmup_runs):
                frame_gpu = torch.from_numpy(self.input_frame).half().cuda()
                if self.tiled:
                    output = tile_processor.process_frame(frame_gpu, infer)
                else:
                    output = infer(frame_gpu)
                _ = output.cpu().numpy()

            torch.cuda.synchronize()

            # Actual Timing
            start_time = time.time()
            for _ in range(self.iterations):
                frame_gpu = torch.from_numpy(self.input_frame).half().cuda()
                if self.tiled:
                    output = tile_processor.process_frame(frame_gpu, infer)
                else:
                    output = infer(frame_gpu)
                _ = output.cpu().numpy()

            torch.cuda.synchronize()
            end_time = time.time()

        total_time = end_time - start_time
        avg_time_ms = (total_time / self.iterations) * 1000

        print("-" * 30)
        print(f"Total time for {self.iterations} runs: {total_time:.4f}s")
        print(f"Average Inference Time: {avg_time_ms:.2f} ms")
        print(f"Throughput: {1.0 / (avg_time_ms / 1000):.2f} FPS")
        print("-" * 30)

        return f"{1.0 / (avg_time_ms / 1000):.2f}"
