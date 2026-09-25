import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ..audio import AudioTrack
from ..scaling import choose_auto_scale, get_screen_size


class Letterboxer:
    """Fits ``display`` (H,W,3) inside ``screen_size`` (h, w) preserving its aspect ratio, centered
    on black bars."""

    def __init__(self):
        self._canvas = None
        self._geom = None  # (nh, nw, sh, sw, dtype) the cached canvas layout is valid for

    def render(self, display, screen_size):
        sh, sw = screen_size
        h, w = display.shape[:2]
        scale = min(sw / w, sh / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        if (nw, nh) != (w, h):
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            display = cv2.resize(display, (nw, nh), interpolation=interp)
        if (nw, nh) == (sw, sh):
            return display  # exact fit, no bars, no canvas needed

        y0, x0 = (sh - nh) // 2, (sw - nw) // 2
        geom = (nh, nw, sh, sw, display.dtype)
        if geom != self._geom or self._canvas is None:
            self._canvas = np.zeros((sh, sw, 3), dtype=display.dtype)
            self._geom = geom
        canvas = self._canvas
        # Clear only the bars (the video region is fully overwritten below); this also removes the
        # FPS / time overlays the caller drew onto the bars on the previous frame.
        if y0 > 0:
            canvas[:y0].fill(0)
            canvas[y0 + nh:].fill(0)
        if x0 > 0:
            canvas[y0:y0 + nh, :x0].fill(0)
            canvas[y0:y0 + nh, x0 + nw:].fill(0)
        canvas[y0:y0 + nh, x0:x0 + nw] = display
        return canvas


def format_time(seconds: float) -> str:
    """Seconds -> M:SS"""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


# cv2.waitKeyEx extended key codes (Windows)
KEY_ESC = 27
KEY_SPACE = 32
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_UP = 2490368
KEY_DOWN = 2621440


class _BaseCv2Player:
    """
    Shared cv2-based SR player: play/pause, seeking, fullscreen toggle, the FPS/time overlay and
    the real-time frame-skipping display loop. Subclasses only supply where frames come from
    (``_make_reader``) and how a source frame becomes a BGR image to show (``_produce_display``).
    """

    config_desc = "SR video player"

    def __init__(self, video_path,
                 fps: float, frame_count: int, frame_size: Tuple[int, int],
                 target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True,
                 seek_step_s: float = 1.0, seek_step_large_s: float = 10.0):
        self.video_path = Path(video_path)
        self.window_name = "SR Video"
        self.seek_step_s = seek_step_s
        self.seek_step_large_s = seek_step_large_s
        self.target_size = target_size
        self.log_prefix = "[videoplayer]"

        self.fps = fps
        self.frame_count = frame_count
        self.frame_size = frame_size

        self.upscale_fn = None
        self.paused = False
        self.fullscreen = start_fullscreen
        self.screen_size = get_screen_size()
        self._reader = None
        self._letterboxer = Letterboxer()
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

    def _make_reader(self):
        """Create and return the background reader (interface: get_latest / request_seek /
        set_paused / stop / is_alive / start / join)."""
        raise NotImplementedError

    def _produce_display(self, frame):
        """Turn one source frame into a BGR (H,W,3) uint8 numpy image ready for cv2.imshow."""
        raise NotImplementedError

    def _release_source(self):
        """Release any decode resource on teardown (no-op unless a subclass owns one)."""

    def _seek(self, delta_seconds: float, current: float):
        target = int(current + delta_seconds * self.fps)
        target = max(0, min(target, max(0, self.frame_count - 1)))
        self._reader.request_seek(target)
        if self.audio is not None:
            self.audio.seek(target / self.fps)

    def _window_closed(self) -> bool:
        """cv2 doesn't fire an event when the user clicks the window's X button, so poll
        WND_PROP_VISIBLE instead."""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)

    def _handle_key(self, key: int, current: float) -> bool:
        """Returns False when playback should stop."""
        if key in (KEY_ESC, ord('q')):
            return False
        if key == KEY_SPACE:
            self.paused = not self.paused
            self._reader.set_paused(self.paused)
            if self.audio is not None:
                self.audio.set_paused(self.paused)
        elif key == KEY_LEFT:
            self._seek(-self.seek_step_s, current)
        elif key == KEY_RIGHT:
            self._seek(self.seek_step_s, current)
        elif key == KEY_UP:
            self._seek(self.seek_step_large_s, current)
        elif key == KEY_DOWN:
            self._seek(-self.seek_step_large_s, current)
        elif key == ord('f'):
            self._toggle_fullscreen()
        return True

    def _print_controls(self):
        print("Controls:")
        print("  Space       Pause / play")
        print(f"  Left/Right  Seek -{self.seek_step_s:g}s / +{self.seek_step_s:g}s")
        print(f"  Up/Down     Seek +{self.seek_step_large_s:g}s / -{self.seek_step_large_s:g}s")
        print("  f           Toggle fullscreen")
        print("  q / Esc     Quit")
        if self.audio is not None and not self.audio.available:
            print("Audio: no track / ffmpeg / sounddevice available, running silent.")

    def play(self):
        if self.upscale_fn is None:
            return

        print(f"[videoplayer] Running config: {self.config_desc}")
        self._print_controls()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif self.target_size is not None:
            cv2.resizeWindow(self.window_name, self.target_size[1], self.target_size[0])

        self._reader = self._make_reader()
        self._reader.start()
        if self.audio is not None:
            self.audio.start()
            self.audio.set_paused(False)

        frame_times = deque(maxlen=20)  # compute FPS (SR + display)
        loop_times = deque(maxlen=20)  # displayed FPS
        last_frame_counter = -1
        last_output = None
        prev_paused = self.paused
        needs_redraw = False
        running = True

        while running:
            frame, frame_counter, pos = self._reader.get_latest()

            if frame is None:
                key = cv2.waitKeyEx(10)
                running = self._handle_key(key, pos) and not self._window_closed()
                if not self._reader.is_alive() and frame is None:
                    break
                continue

            if frame_counter != last_frame_counter or last_output is None:
                t0 = time.perf_counter()
                output = self._produce_display(frame)
                frame_times.append((time.perf_counter() - t0) * 1000)

                last_output = output
                last_frame_counter = frame_counter

                if not self.paused:
                    loop_times.append(time.perf_counter())
                needs_redraw = True

            # A pause toggle changes the overlay (PAUSED text) so force a one-off redraw
            if self.paused != prev_paused:
                needs_redraw = True
                prev_paused = self.paused

            # Idle: same frame, same state. Just keep the window responsive instead of copying
            if not needs_redraw:
                key = cv2.waitKeyEx(1)
                running = self._handle_key(key, pos) and not self._window_closed()
                continue

            display = last_output.copy()

            if self.paused:
                loop_times.clear()

            if self.fullscreen:
                display = self._letterboxer.render(display, self.screen_size)

            # overlay

            # compute FPS = model + display-convert cost only
            # achieved FPS = real rate of new frames
            compute_fps = 1000.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            achieved_fps = ((len(loop_times) - 1) / (loop_times[-1] - loop_times[0])
                            if len(loop_times) >= 2 and loop_times[-1] > loop_times[0] else 0.0)
            if self.paused:
                lines = ["PAUSED"]
            else:
                lines = [f"{achieved_fps:.0f} FPS (displayed)", f"{compute_fps:.0f} FPS (GPU)"]
            for i, line in enumerate(lines):
                cv2.putText(display, line, (20, 40 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            pos_s = pos / self.fps
            dur_s = self.frame_count / self.fps
            time_text = f"{format_time(pos_s)} / {format_time(dur_s)}"
            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            (text_w, text_h), _ = cv2.getTextSize(time_text, font, font_scale, thickness)
            margin = 20
            text_x = display.shape[1] - text_w - margin
            text_y = display.shape[0] - margin
            cv2.putText(display, time_text, (text_x, text_y), font, font_scale, (160, 160, 160), thickness)

            cv2.imshow(self.window_name, display)

            needs_redraw = False
            key = cv2.waitKeyEx(1)
            running = self._handle_key(key, pos) and not self._window_closed()

        # end of the loop
        self._reader.stop()
        self._reader.join(timeout=1.0)
        if self.audio is not None:
            self.audio.stop()
        self._release_source()
        cv2.destroyAllWindows()
