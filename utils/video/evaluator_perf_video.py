from pathlib import Path
from typing import Literal, TypeAlias
import numpy as np
import torch
import time

from utils.video.export import export_onnx, export_trt
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

    def __init__(self, model, name, runtype: Runtype, image_size=(720, 1280), upscale_factor=2, tiled=False,
                 tile_size=256, dont_export=False, warmup_runs=10, iterations=100):
        self.model = model
        self.model.eval()
        self.model.half()
        self.name = name

        self.upscale_factor = upscale_factor
        self.tiled = tiled
        self.tile_size = tile_size
        self.image_size = (image_size[0], image_size[1], 3)
        self.input_frame = np.random.randint(0, 255, self.image_size)
        self.input_size = (tile_size, tile_size, 3) if tiled else self.image_size

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
        output_path = Path(f"exports/onnx/{self.name}_{self.input_size[0]}x{self.input_size[1]}_cv2.onnx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.dont_export:
            export_onnx(VideoWrapperCV2(self.model), output_path, self.input_size)

        if self.runtype == "onnxruntime-tensorrt":
            import torch_tensorrt
            providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': True})]
        elif self.runtype == "onnxruntime-cuda":
            providers = ['CUDAExecutionProvider']
        elif self.runtype == "onnxruntime-openvino":
            providers = [('OpenVINOExecutionProvider', {"device_type": "GPU", "precision": "FP16"})]

        # Load model
        ort_session = ort.InferenceSession(output_path, providers=providers)

        # Inference
        print(f"Using input shape: {self.input_size}")

        # Define callback for model inference
        def infer(tile):
            return ort_session.run(None, {"input": tile.astype(np.float16)})[0]

        # Define callback for upscaling the frame
        if self.tiled:
            tile_processor = TileProcessor(upscale_factor=self.upscale_factor, tile_size=self.tile_size, overlap=8)

            def upscale(frame):
                return tile_processor.process_frame(frame, infer)
        else:
            def upscale(frame):
                return infer(frame)

        return self._measuring_loop(upscale)

    # ============TENSORRT=============

    def evaluate_tensorrt(self):
        import torch_tensorrt

        torch.cuda.empty_cache()

        output_path = Path(f"exports/trt/{self.name}_{self.input_size[0]}x{self.input_size[1]}_cv2.pt2")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.dont_export:
            model = export_trt(VideoWrapperCV2(self.model), output_path, self.input_size)
        else:
            model = torch.export.load(output_path).module()

        # Inference
        print(f"Using input shape: {self.input_size}")

        # Define callback for model inference
        def infer(tile):
            return model(tile)

        # Define callback for upscaling the frame
        if self.tiled:
            tile_processor = TileProcessorTorch(self.upscale_factor, self.tile_size, overlap=8)

            def upscale(frame):
                frame_gpu = torch.from_numpy(frame).half().cuda()
                return tile_processor.process_frame(frame_gpu, infer).cpu().numpy()

        else:
            def upscale(frame):
                frame_gpu = torch.from_numpy(frame).half().cuda()
                return infer(frame_gpu).cpu().numpy()

        return self._measuring_loop(upscale)

    #########################

    def _measuring_loop(self, infer_fn):
        # Warmup
        for _ in range(self.warmup_runs):
            infer_fn(self.input_frame)

        torch.cuda.synchronize()

        # Actual Timing
        start_time = time.time()
        for _ in range(self.iterations):
            infer_fn(self.input_frame)

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
