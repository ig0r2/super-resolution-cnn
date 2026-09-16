import gc
import time
from typing import Literal, TypeAlias

import numpy as np
import torch

from utils.path import get_project_root
from utils.video.export import export_onnx, export_trt
from utils.video.export_trt_engine import (export_onnx_raw, get_raw_trt_engine, TRTRawRunner,
                                           estimate_conv_workspace_bytes)
from utils.video.model_utils import TileProcessor, TileProcessorTorch

Runtype: TypeAlias = Literal[
    'tensorrt', 'tensorrt-pt2', 'onnxruntime-cuda', 'onnxruntime-tensorrt', 'onnxruntime-openvino', 'onnxruntime-directml']


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
                 tile_size=256, warmup_runs=10, iterations=100, use_fp32=False):
        self.use_fp32 = use_fp32
        self.torch_dtype = torch.float32 if use_fp32 else torch.float16
        self.np_dtype = np.float32 if use_fp32 else np.float16
        self.precision_suffix = "_fp32" if use_fp32 else ""

        self.model = model
        self.model.eval()
        self.model.to(self.torch_dtype)
        self.name = name
        print(f"Using {'FP32' if use_fp32 else 'FP16'}")

        self.upscale_factor = upscale_factor
        self.tiled = tiled
        self.tile_size = tile_size
        self.image_size = (image_size[0], image_size[1], 3)
        self.input_frame = np.random.randint(0, 255, self.image_size)
        self.input_size = (tile_size, tile_size, 3) if tiled else self.image_size

        self.warmup_runs = warmup_runs
        self.iterations = iterations
        self.runtype: Runtype = runtype

    def evaluate(self):
        if self.runtype == 'tensorrt':
            return self.evaluate_tensorrt_raw()
        elif self.runtype == 'tensorrt-pt2':
            return self.evaluate_tensorrt_pt2()
        elif self.runtype.startswith('onnxruntime'):
            return self.evaluate_onnx()

    # ===================ONNX================

    def evaluate_onnx(self):
        import onnxruntime as ort

        # Export model
        output_path = get_project_root(
            f"exports/onnx/{self.name}_{self.input_size[0]}x{self.input_size[1]}_{self.upscale_factor}x_cv2"
            f"{self.precision_suffix}.onnx")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists():
            export_onnx(VideoWrapperCV2(self.model), output_path, self.input_size, use_fp32=self.use_fp32)

        if self.runtype == "onnxruntime-tensorrt":
            providers = [('TensorrtExecutionProvider', {'trt_fp16_enable': not self.use_fp32})]
        elif self.runtype == "onnxruntime-cuda":
            providers = ['CUDAExecutionProvider']
        elif self.runtype == "onnxruntime-openvino":
            providers = [('OpenVINOExecutionProvider',
                          {"device_type": "GPU", "precision": "FP32" if self.use_fp32 else "FP16"})]
        elif self.runtype == "onnxruntime-directml":
            providers = ['DmlExecutionProvider']

        # Load model
        ort_session = ort.InferenceSession(output_path, providers=providers)

        # Inference
        print(f"Using input shape: {self.input_size}")

        # Define callback for model inference
        def infer(tile):
            return ort_session.run(None, {"input": tile.astype(self.np_dtype)})[0]

        # Define callback for upscaling the frame
        if self.tiled:
            tile_processor = TileProcessor(upscale_factor=self.upscale_factor, tile_size=self.tile_size, overlap=8)

            def upscale(frame):
                return tile_processor.process_frame(frame, infer)
        else:
            def upscale(frame):
                return infer(frame)

        try:
            return self._measuring_loop(upscale)
        finally:
            del ort_session, infer, upscale
            if self.tiled: del tile_processor
            self._free_gpu_memory()

    # ============TENSORRT (raw)=============

    def evaluate_tensorrt_raw(self):
        torch.cuda.empty_cache()

        build_dir = get_project_root("exports/trt_raw")
        build_dir.mkdir(parents=True, exist_ok=True)
        base_tag = (f"{self.name}_{self.input_size[0]}x{self.input_size[1]}_{self.upscale_factor}x_cv2"
                    f"{self.precision_suffix}")
        onnx_path = build_dir / f"{base_tag}.onnx"
        engine_path = build_dir / f"{base_tag}.engine"

        if not engine_path.exists():
            label = (f"{self.name} at {self.input_size[:2]} "
                     f"(tiled={self.tiled}, tile_size={self.tile_size if self.tiled else None})")
            estimate_conv_workspace_bytes(VideoWrapperCV2(self.model), self.input_size[:2],
                                           use_fp32=self.use_fp32, label=label)

            export_onnx_raw(VideoWrapperCV2(self.model), onnx_path, self.input_size[:2], use_fp32=self.use_fp32)

        engine = get_raw_trt_engine(onnx_path, engine_path, use_fp32=self.use_fp32, opt_level=1)
        runner = TRTRawRunner(engine)

        # Inference
        print(f"Using input shape: {self.input_size}")

        # Define callback for model inference
        def infer(tile):
            return runner(tile)

        # Define callback for upscaling the frame
        if self.tiled:
            tile_processor = TileProcessorTorch(self.upscale_factor, self.tile_size, overlap=8,
                                                dtype=self.torch_dtype)

            def upscale(frame):
                frame_gpu = torch.from_numpy(frame).to(self.torch_dtype).cuda()
                return tile_processor.process_frame(frame_gpu, infer).cpu().numpy()

        else:
            def upscale(frame):
                frame_gpu = torch.from_numpy(frame).to(self.torch_dtype).cuda()
                return infer(frame_gpu).cpu().numpy()

        try:
            return self._measuring_loop(upscale)
        finally:
            del engine, runner, infer, upscale
            if self.tiled:
                del tile_processor
            self._free_gpu_memory()

    # ============TENSORRT (torch_tensorrt .pt2)=============

    def evaluate_tensorrt_pt2(self):

        torch.cuda.empty_cache()

        output_path = get_project_root(
            f"exports/trt/{self.name}_{self.input_size[0]}x{self.input_size[1]}_{self.upscale_factor}x_cv2"
            f"{self.precision_suffix}.pt2")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists():
            model = export_trt(VideoWrapperCV2(self.model), output_path, self.input_size, use_fp32=self.use_fp32)
        else:
            import torch_tensorrt
            model = torch_tensorrt.load(output_path).module()

        # Inference
        print(f"Using input shape: {self.input_size}")

        # Define callback for model inference
        def infer(tile):
            return model(tile)

        # Define callback for upscaling the frame
        if self.tiled:
            tile_processor = TileProcessorTorch(self.upscale_factor, self.tile_size, overlap=8,
                                                dtype=self.torch_dtype)

            def upscale(frame):
                frame_gpu = torch.from_numpy(frame).to(self.torch_dtype).cuda()
                return tile_processor.process_frame(frame_gpu, infer).cpu().numpy()

        else:
            def upscale(frame):
                frame_gpu = torch.from_numpy(frame).to(self.torch_dtype).cuda()
                return infer(frame_gpu).cpu().numpy()

        try:
            return self._measuring_loop(upscale)
        finally:
            del model, infer, upscale
            if self.tiled:
                del tile_processor
            self._free_gpu_memory()

    #########################

    def _free_gpu_memory(self):
        """Explicitly drop the model and any GPU-resident state, then reclaim VRAM."""
        self.model = self.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

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
