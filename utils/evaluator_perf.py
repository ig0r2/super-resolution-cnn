import torch
import time


class EvaluatorPerf:
    """
    Evaluacija brzine inferense. Kroz model se pusta random tenzor odredjen broj iteracija i racuna kao FPS
    """

    def __init__(self, model, image_size=(720, 1280), warmup_runs=10, iterations=100, use_half=True):
        self.model = model
        self.input_size = (1, 3, image_size[0], image_size[1])
        self.warmup_runs = warmup_runs
        self.iterations = iterations
        self.use_half = use_half

    def evaluate(self):
        assert torch.cuda.is_available()
        device = torch.device("cuda")

        torch.cuda.empty_cache()

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            print("Using cuDNN benchmark mode")

        self.model.eval()
        self.model = torch.compile(self.model)
        input_tensor = torch.randn(self.input_size, device=device)

        if self.use_half:
            self.model = self.model.half()
            input_tensor = input_tensor.half()
            print(f"Using float16")
        else:
            print(f"Using float32")

        print(f"Using input shape: {self.input_size}")

        with torch.no_grad():
            # Warmup
            for _ in range(self.warmup_runs):
                _ = self.model(input_tensor)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Actual Timing
            start_time = time.time()
            for _ in range(self.iterations):
                _ = self.model(input_tensor)

            torch.cuda.synchronize()
            end_time = time.time()

            peak_vram_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
            peak_vram_res = torch.cuda.max_memory_reserved() / 1024 ** 2

        total_time = end_time - start_time
        avg_time_ms = (total_time / self.iterations) * 1000

        print("-" * 30)
        print(f"Total time for {self.iterations} runs: {total_time:.4f}s")
        print(f"Average Inference Time: {avg_time_ms:.2f} ms")
        print(f"Throughput: {1.0 / (avg_time_ms / 1000):.2f} FPS")
        print("-" * 30)
        print(f"Peak VRAM Allocated: {peak_vram_alloc:.2f} MB")
        print(f"Peak VRAM Reserved: {peak_vram_res:.2f} MB")
        print("-" * 30)

        return f"{1.0 / (avg_time_ms / 1000):.2f}", f"{int(peak_vram_res)}"
