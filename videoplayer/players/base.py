from pathlib import Path
from typing import Optional, Tuple

from ..audio import AudioTrack
from ..scaling import choose_auto_scale, get_screen_size


def format_time(seconds: float) -> str:
    """Seconds -> M:SS"""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


class _BasePlayer:
    """
    Display-independent part of an SR player: the decoder, audio track, frame geometry, the SR
    function, scale selection and seeking. _BaseCv2Player and _BaseGlPlayer add the display loop.
    """

    config_desc = "SR video player"

    def __init__(self, video_path, decoder,
                 target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True,
                 seek_step_s: float = 1.0, seek_step_large_s: float = 10.0):
        self.video_path = Path(video_path)
        self.window_name = "SR Video"
        self.seek_step_s = seek_step_s
        self.seek_step_large_s = seek_step_large_s
        self.target_size = target_size
        self.log_prefix = "[videoplayer]"

        self.decoder = decoder
        self.fps = decoder.fps
        self.frame_count = len(decoder)
        self.frame_size = (decoder.height, decoder.width)

        self.upscale_fn = None
        self.paused = False
        self.fullscreen = start_fullscreen
        self.screen_size = get_screen_size()
        self._reader = None
        self.audio = AudioTrack(self.video_path) if enable_audio else None

    @property
    def size(self):
        return self.frame_size

    def set_upscale_fn(self, upscale_fn):
        self.upscale_fn = upscale_fn
        return self

    def configure_scale(self, candidate_scales=(2, 3, 4)) -> int:
        """Pick the model scale for this video on the current screen."""
        decision = choose_auto_scale(self.frame_size, self.screen_size, candidate_scales)
        print(f"{self.log_prefix} Frame {self.frame_size} | Screen {self.screen_size} | "
              f"Model scale {decision.model_scale}x -> {decision.model_output_size} | "
              f"Target (bicubic) {decision.target_size}")
        self.target_size = decision.target_size if decision.target_size != decision.model_output_size else None
        return decision.model_scale

    def play(self):
        raise NotImplementedError

    def _release_source(self):
        self.decoder.close()

    def _seek(self, delta_seconds: float, current: float):
        target = int(current + delta_seconds * self.fps)
        target = max(0, min(target, max(0, self.frame_count - 1)))
        self._reader.request_seek(target)
        if self.audio is not None:
            self.audio.seek(target / self.fps)

    def _print_controls(self):
        print("Controls:")
        print("  Space       Pause / play")
        print(f"  Left/Right  Seek -{self.seek_step_s:g}s / +{self.seek_step_s:g}s")
        print(f"  Up/Down     Seek +{self.seek_step_large_s:g}s / -{self.seek_step_large_s:g}s")
        print("  f           Toggle fullscreen")
        print("  q / Esc     Quit")
        if self.audio is not None and not self.audio.available:
            print("Audio: no track / PyAV / sounddevice available, running silent.")
