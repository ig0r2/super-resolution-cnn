import numpy as np
import torch


def _compute_coords(tile_size, overlap, h, w, scale):
    """Izracunava koordinate za svaki tile"""
    step = tile_size - overlap
    coords = []

    for y in range(0, h, step):
        y2 = min(y + tile_size, h)
        y1 = max(0, y2 - tile_size)
        ys1, ys2 = y1 * scale, y2 * scale

        for x in range(0, w, step):
            x2 = min(x + tile_size, w)
            x1 = max(0, x2 - tile_size)
            xs1, xs2 = x1 * scale, x2 * scale
            coords.append((y1, y2, x1, x2, ys1, ys2, xs1, xs2))

    return coords


class TileProcessor:
    """Sadrzi logiku za ineferencu preko delova - preko numpy (za onnxruntime)"""

    def __init__(self, upscale_factor, tile_size=256, overlap=8):
        self.initialized = False
        self.upscale_factor = upscale_factor
        self.tile_size = tile_size
        self.overlap = overlap

    def init(self, frame):
        h, w, c = frame.shape
        self.coords = _compute_coords(self.tile_size, self.overlap, h, w, self.upscale_factor)

        # Preallocate buffers
        scaled_h = h * self.upscale_factor
        scaled_w = w * self.upscale_factor
        # Accumulation buffers (float32 for precision during averaging)
        self.output_acc = np.zeros((scaled_h, scaled_w, c), dtype=np.float32)
        self.count_acc = np.zeros((scaled_h, scaled_w, 1), dtype=np.uint8)
        # Final output buffer (uint8)
        self.result = np.empty((scaled_h, scaled_w, c), dtype=np.uint8)

    def process_frame(self, frame, infer_fn):
        """
        Process frame through tiled inference.

        Args:
            frame: Input image (H, W, C)
            infer_fn: Function that takes tile and returns upscaled tile

        Returns:
            Upscaled image (H*scale, W*scale, C) as uint8
        """
        # If first pass, allocate buffers
        if not self.initialized:
            self.init(frame)
            self.initialized = True

        # Reset buffers
        self.output_acc.fill(0.0)
        self.count_acc.fill(0)

        # Process all tiles using precomputed coordinates
        for y1, y2, x1, x2, ys1, ys2, xs1, xs2 in self.coords:
            tile = frame[y1:y2, x1:x2, :]
            pred = infer_fn(tile)

            self.output_acc[ys1:ys2, xs1:xs2, :] += pred
            self.count_acc[ys1:ys2, xs1:xs2, :] += 1

        # Average and convert to uint8 in-place
        np.divide(self.output_acc, self.count_acc, out=self.output_acc)
        np.clip(self.output_acc, 0, 255, out=self.output_acc)
        # Copy to result buffer with type conversion
        self.result[:] = self.output_acc

        return self.result


class TileProcessorTorch:
    """Sadrzi logiku za ineferencu preko delova - preko torch (za tensorrt)"""

    def __init__(self, upscale_factor, tile_size=256, overlap=8):
        self.initialized = False
        self.upscale_factor = upscale_factor
        self.tile_size = tile_size
        self.overlap = overlap

    def init(self, frame):
        h, w, c = frame.shape
        self.coords = _compute_coords(self.tile_size, self.overlap, h, w, self.upscale_factor)
        scaled_h = h * self.upscale_factor
        scaled_w = w * self.upscale_factor
        self.output_acc = torch.zeros((scaled_h, scaled_w, c), dtype=torch.float16, device="cuda")
        self.count_acc = torch.zeros((scaled_h, scaled_w, 1), dtype=torch.float16, device="cuda")
        self.result = torch.empty((scaled_h, scaled_w, c), dtype=torch.uint8, device="cuda")

    def process_frame(self, frame, infer_fn):
        if not self.initialized:
            self.init(frame)
            self.initialized = True

        self.output_acc.zero_()
        self.count_acc.zero_()

        for y1, y2, x1, x2, ys1, ys2, xs1, xs2 in self.coords:
            tile = frame[y1:y2, x1:x2, :]
            pred = infer_fn(tile)

            self.output_acc[ys1:ys2, xs1:xs2, :].add_(pred)
            self.count_acc[ys1:ys2, xs1:xs2, :].add_(1)

        torch.div(self.output_acc, self.count_acc, out=self.output_acc)
        torch.clamp(self.output_acc, 0, 255, out=self.output_acc)
        # Copy to uint8 buffer
        self.result.copy_(self.output_acc.to(torch.uint8))

        return self.result
