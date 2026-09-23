import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .audio import AudioTrack
from .scaling import choose_auto_scale, get_screen_size


def letterbox_to_screen(display, screen_size):
    """Fit ``display`` (H,W,3) inside ``screen_size`` (h, w) preserving its aspect ratio, centered
    on a black canvas. Mirrors the GL players' letterboxing so fullscreen keeps the video ratio
    instead of stretching to fill the screen."""
    sh, sw = screen_size
    h, w = display.shape[:2]
    scale = min(sw / w, sh / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if (nw, nh) != (w, h):
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        display = cv2.resize(display, (nw, nh), interpolation=interp)
    if (nw, nh) == (sw, sh):
        return display
    canvas = np.zeros((sh, sw, 3), dtype=display.dtype)
    y0, x0 = (sh - nh) // 2, (sw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = display
    return canvas


def format_time(seconds: float) -> str:
    """Seconds -> M:SS (e.g. 153 -> '2:33')."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


# cv2.waitKeyEx extended key codes (Windows)
KEY_ESC = 27
KEY_SPACE = 32
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_UP = 2490368
KEY_DOWN = 2621440


class _FrameReader(threading.Thread):
    """
    Reads frames from a cv2.VideoCapture at real-time pace on a background thread and always
    exposes only the latest one, like utils/video/videostream.py's queue(maxsize=1). This is
    what lets the main loop skip frames instead of slowing down when upscale_fn is slower than
    the video's native framerate.
    """

    def __init__(self, cap: cv2.VideoCapture, fps: float):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0

        self.lock = threading.Lock()
        self.frame = None
        self.pos_frames = 0.0
        self.frame_id = 0

        self.running = True
        self.paused = False
        self._seek_request: Optional[float] = None

    def run(self):
        next_time = time.perf_counter()

        while self.running:
            if self.paused:
                time.sleep(0.01)
                next_time = time.perf_counter()
                continue

            with self.lock:
                seek_to = self._seek_request
                self._seek_request = None
            if seek_to is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                next_time = time.perf_counter()

            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            with self.lock:
                self.frame = frame
                self.pos_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.frame_id += 1

            next_time += self.frame_time
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()  # fell behind, don't try to catch up

    def get_latest(self):
        with self.lock:
            return self.frame, self.frame_id, self.pos_frames

    def request_seek(self, target_frame: float):
        with self.lock:
            self._seek_request = target_frame

    def set_paused(self, paused: bool):
        self.paused = paused

    def stop(self):
        self.running = False


class VideoPlayer:
    """
    OpenCV based video player with play/pause, seeking, and a maximize/fullscreen toggle.

    upscale_fn is applied to the latest available native-resolution BGR frame; if target_size
    is given, the upscaled frame is bicubic-downscaled to that exact (h, w) so it fits the
    screen. Frames are read on a background thread at real-time pace, so if upscale_fn is
    slower than the video's native framerate, frames are skipped rather than played back slow.
    """

    def __init__(self, video_path, window_name: str = "SR Video", show_original: bool = False,
                 seek_step_s: float = 1.0, seek_step_large_s: float = 10.0,
                 target_size: Optional[Tuple[int, int]] = None,
                 enable_audio: bool = True, start_fullscreen: bool = True):
        self.video_path = Path(video_path)
        self.window_name = window_name
        self.original_window_name = "Original Video"
        self.show_original = show_original
        self.seek_step_s = seek_step_s
        self.seek_step_large_s = seek_step_large_s
        self.target_size = target_size

        print(f"[videoplayer] Opening video {self.video_path.name} ...")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

        self.upscale_fn = None
        self.paused = False
        self.fullscreen = start_fullscreen
        self.screen_size: Optional[Tuple[int, int]] = None
        self._reader: Optional[_FrameReader] = None
        self.audio = AudioTrack(self.video_path) if enable_audio else None

    @property
    def size(self):
        return self.frame_size

    def set_upscale_fn(self, upscale_fn):
        self.upscale_fn = upscale_fn
        return self

    def configure_scale(self, candidate_scales=(2, 3, 4)) -> int:
        """Pick the model scale for this video on the current screen, set the display target_size,
        and return the chosen integer scale (which drives the SR model / engine)."""
        screen_size = get_screen_size()
        self.screen_size = screen_size
        decision = choose_auto_scale(self.frame_size, screen_size, candidate_scales)
        print(f"[videoplayer] Frame {self.frame_size} | Screen {screen_size} | "
              f"Model scale {decision.model_scale}x -> {decision.model_output_size} | "
              f"Target (bicubic) {decision.target_size}")
        self.target_size = decision.target_size if decision.target_size != decision.model_output_size else None
        return decision.model_scale

    def _seek(self, delta_seconds: float, current_pos_frames: float):
        target = current_pos_frames + delta_seconds * self.fps
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

    def _handle_key(self, key: int, current_pos_frames: float) -> bool:
        """Returns False when playback should stop."""
        if key in (KEY_ESC, ord('q')):
            return False
        if key == KEY_SPACE:
            self.paused = not self.paused
            self._reader.set_paused(self.paused)
            if self.audio is not None:
                self.audio.set_paused(self.paused)
        elif key == KEY_LEFT:
            self._seek(-self.seek_step_s, current_pos_frames)
        elif key == KEY_RIGHT:
            self._seek(self.seek_step_s, current_pos_frames)
        elif key == KEY_UP:
            self._seek(self.seek_step_large_s, current_pos_frames)
        elif key == KEY_DOWN:
            self._seek(-self.seek_step_large_s, current_pos_frames)
        elif key in (ord('f'), ord('m')):
            self._toggle_fullscreen()
        return True

    def _print_controls(self):
        print("Controls:")
        print("  Space       Pause / play")
        print(f"  Left/Right  Seek -{self.seek_step_s:g}s / +{self.seek_step_s:g}s")
        print(f"  Up/Down     Seek +{self.seek_step_large_s:g}s / -{self.seek_step_large_s:g}s")
        print("  f / m       Toggle fullscreen / maximize")
        print("  q / Esc     Quit")
        if self.audio is not None and not self.audio.available:
            print("Audio: no track / ffmpeg / sounddevice available, running silent.")

    def play(self):
        if self.upscale_fn is None:
            return

        self._print_controls()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif self.target_size is not None:
            cv2.resizeWindow(self.window_name, self.target_size[1], self.target_size[0])
        if self.show_original:
            cv2.namedWindow(self.original_window_name, cv2.WINDOW_NORMAL)

        self._reader = _FrameReader(self.cap, self.fps)
        self._reader.start()
        if self.audio is not None:
            self.audio.start()
            self.audio.set_paused(False)

        frame_times = deque(maxlen=20)  # compute cost per new frame (SR + display resize)
        loop_times = deque(maxlen=30)  # wall-clock timestamps of new frames -> achieved FPS
        last_frame_id = -1
        last_output = None
        prev_paused = self.paused
        needs_redraw = False
        running = True

        while running:
            frame, frame_id, pos_frames = self._reader.get_latest()

            if frame is None:
                key = cv2.waitKeyEx(10)
                running = self._handle_key(key, pos_frames) and not self._window_closed()
                if not self._reader.is_alive() and frame is None:
                    break
                continue

            if frame_id != last_frame_id or last_output is None:
                t0 = time.perf_counter()
                output = self.upscale_fn(frame)

                if self.target_size is not None:
                    th, tw = self.target_size
                    output = cv2.resize(output, (tw, th), interpolation=cv2.INTER_CUBIC)

                frame_time_ms = (time.perf_counter() - t0) * 1000
                frame_times.append(frame_time_ms)

                last_output = output
                last_frame_id = frame_id

                if not self.paused:
                    loop_times.append(time.perf_counter())
                needs_redraw = True

            # A pause toggle changes the overlay (PAUSED text) force a one-off redraw
            if self.paused != prev_paused:
                needs_redraw = True
                prev_paused = self.paused

            # Idle: same frame, same state. Just keep the window responsive instead of copying,
            # re-drawing overlays and re-showing the identical image hundreds of times a second.
            if not needs_redraw:
                key = cv2.waitKeyEx(1)
                running = self._handle_key(key, pos_frames) and not self._window_closed()
                continue

            display = last_output.copy()

            if self.paused:
                loop_times.clear()  # avoid a stale gap spanning the pause on resume

            # Letterbox first so the overlays below are drawn onto the final screen-sized canvas;
            # anchored to its corners, they land on the black bars instead of over the video.
            if self.fullscreen:
                if self.screen_size is None:
                    self.screen_size = get_screen_size()
                display = letterbox_to_screen(display, self.screen_size)

            # compute FPS = model + display-resize cost only (decode is on the reader thread, and
            # GUI/pacing are excluded) -> an upper bound. achieved FPS = real rate of new frames
            # reaching the screen -> includes decode wait, GUI, and the real-time pacing cap.
            compute_fps = 1000.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            achieved_fps = ((len(loop_times) - 1) / (loop_times[-1] - loop_times[0])
                            if len(loop_times) >= 2 and loop_times[-1] > loop_times[0] else 0.0)
            if self.paused:
                lines = ["PAUSED"]
            else:
                lines = [f"{achieved_fps:.1f} FPS (displayed)", f"{compute_fps:.1f} FPS (GPU)"]
            for i, line in enumerate(lines):
                cv2.putText(display, line, (20, 40 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            pos_s = pos_frames / self.fps
            dur_s = self.frame_count / self.fps
            time_text = f"{format_time(pos_s)} / {format_time(dur_s)}"
            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            (text_w, text_h), _ = cv2.getTextSize(time_text, font, font_scale, thickness)
            margin = 20
            text_x = display.shape[1] - text_w - margin
            text_y = display.shape[0] - margin
            cv2.putText(display, time_text, (text_x, text_y), font, font_scale, (160, 160, 160), thickness)

            cv2.imshow(self.window_name, display)
            if self.show_original:
                cv2.imshow(self.original_window_name, frame)

            needs_redraw = False
            key = cv2.waitKeyEx(1)
            running = self._handle_key(key, pos_frames) and not self._window_closed()

        self._reader.stop()
        self._reader.join(timeout=1.0)
        if self.audio is not None:
            self.audio.stop()
        self.cap.release()
        cv2.destroyAllWindows()
