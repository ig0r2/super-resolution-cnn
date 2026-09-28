import time
from collections import deque

import cv2
import numpy as np

from .base import _BasePlayer, format_time
from ..decode.reader import _DecodeReader


class Letterboxer:
    """Centers ``display`` (H,W,3) on black bars filling ``screen_size`` (h, w)."""

    def __init__(self):
        self._canvas = None
        self._geom = None  # (nh, nw, sh, sw, dtype) the cached canvas layout is valid for

    def render(self, display, screen_size):
        sh, sw = screen_size
        nh, nw = display.shape[:2]
        if (nw, nh) == (sw, sh) or nw > sw or nh > sh:
            # exact fit (no bars needed), or larger than the screen
            return display

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


# cv2.waitKeyEx extended key codes (Windows)
KEY_ESC = 27
KEY_SPACE = 32
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_UP = 2490368
KEY_DOWN = 2621440


class _BaseCv2Player(_BasePlayer):
    """
    cv2 display on top of _BasePlayer: cv2 window, fullscreen toggle, keyboard controls, the
    FPS/time overlay (letterboxed in fullscreen) and the real-time frame-skipping display loop.
    Subclasses pick the decoder and supply how a decoded frame becomes a BGR image to show
    (``_produce_display``).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._letterboxer = Letterboxer()

    def _produce_display(self, frame):
        """Turn one source frame into a BGR (H,W,3) uint8 numpy image ready for cv2.imshow."""
        raise NotImplementedError

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

    def play(self):
        if self.upscale_fn is None:
            return

        print(f"[videoplayer] Running config: {self.config_desc}")
        print("[videoplayer] Starting playback.")
        self._print_controls()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif self.target_size is not None:
            cv2.resizeWindow(self.window_name, self.target_size[1], self.target_size[0])

        self._reader = _DecodeReader(self.decoder)
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
