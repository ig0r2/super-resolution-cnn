from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class ScaleDecision:
    model_scale: int
    model_output_size: Tuple[int, int]  # (h, w) produced by the SR model
    target_size: Tuple[int, int]  # (h, w) after bicubic downscale to fit the screen


def get_screen_size() -> Tuple[int, int]:
    """Returns (height, width) of the primary screen."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)
        if width > 0 and height > 0:
            return height, width
    except Exception:
        pass

    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return height, width
    except Exception:
        return 1080, 1920


def choose_auto_scale(frame_size: Tuple[int, int], screen_size: Tuple[int, int],
                       candidate_scales: Iterable[int] = (2, 3, 4)) -> ScaleDecision:
    """
    Picks the smallest integer model scale whose upscaled output is at least as big as the
    screen (in both dimensions, keeping aspect ratio), then computes the exact target size to
    bicubic-downscale that output to so it fits the screen without exceeding it.
    """
    frame_h, frame_w = frame_size
    screen_h, screen_w = screen_size

    fit_scale = min(screen_w / frame_w, screen_h / frame_h)

    candidates = sorted(candidate_scales)
    model_scale = candidates[-1]
    for s in candidates:
        if s >= fit_scale:
            model_scale = s
            break

    model_output_size = (frame_h * model_scale, frame_w * model_scale)

    target_w = max(2, round(frame_w * fit_scale))
    target_h = max(2, round(frame_h * fit_scale))
    target_w -= target_w % 2
    target_h -= target_h % 2

    return ScaleDecision(model_scale, model_output_size, (target_h, target_w))
