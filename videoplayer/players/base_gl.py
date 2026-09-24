import time
from collections import deque

import glfw
import torch

from .base_cv2 import _BaseCv2Player, format_time


class _BaseGlPlayer(_BaseCv2Player):
    """
    Shared glfw/OpenGL play loop for the GPU-display SR players. Reuses _BaseCv2Player (via a
    concrete decode player as the second base) for NVDEC decode / background reader / audio /
    seeking, and swaps the cv2 display loop for a glfw one that shows frames straight from GPU
    memory and puts the FPS / position stats in the window title bar instead of drawing them onto
    the frame (no CPU text-draw pass).

    Subclasses supply only the display surface and how a frame is shown: ``_gl_open`` (create the
    surface and register the key callback), ``_gl_infer`` (run SR on a new frame and upload it) and
    ``_gl_present`` (show the current frame with a title).
    """

    def _gl_ready(self) -> bool:
        """Whether the SR back-end is set up and playback can start."""
        return self.upscale_fn is not None

    def _gl_open(self, on_key):
        """Create the GL surface, register ``on_key``, and return the surface."""
        raise NotImplementedError

    def _gl_infer(self, frame):
        """Run SR on a new source frame and upload/prepare it for display."""
        raise NotImplementedError

    def _gl_present(self, title: str, have_frame: bool):
        """Present the current frame (if any) with ``title`` in the window bar."""
        raise NotImplementedError

    # --- shared loop -----------------------------------------------------------------------------

    def _make_key_callback(self):
        def on_key(window, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                self.paused = not self.paused
                self._reader.set_paused(self.paused)
                if self.audio is not None:
                    self.audio.set_paused(self.paused)
            elif key == glfw.KEY_LEFT:
                self._seek(-self.seek_step_s, self._current_index)
            elif key == glfw.KEY_RIGHT:
                self._seek(self.seek_step_s, self._current_index)
            elif key == glfw.KEY_UP:
                self._seek(self.seek_step_large_s, self._current_index)
            elif key == glfw.KEY_DOWN:
                self._seek(-self.seek_step_large_s, self._current_index)
            elif key == glfw.KEY_F:
                self._surface.toggle_fullscreen()

        return on_key

    def play(self):
        if not self._gl_ready():
            return

        print(f"[videoplayer] Running config: {self.config_desc}")
        self._print_controls()

        self._current_index = 0
        self._surface = self._gl_open(self._make_key_callback())

        self._reader = self._make_reader()
        self._reader.start()
        if self.audio is not None:
            self.audio.start()
            self.audio.set_paused(False)

        frame_times = deque(maxlen=20)
        loop_times = deque(maxlen=20)
        last_frame_id = -1
        have_frame = False
        last_print = 0.0  # throttle console FPS to ~1 Hz (visible in fullscreen)

        while not self._surface.should_close():
            self._surface.poll()
            frame, frame_id, index = self._reader.get_latest()
            self._current_index = index

            if frame is None:
                if not self._reader.is_alive():
                    break
                time.sleep(0.005)
                continue

            if frame_id != last_frame_id:
                t0 = time.perf_counter()
                self._gl_infer(frame)
                torch.cuda.synchronize()  # so the timing reflects real GPU work, not just launch
                frame_times.append((time.perf_counter() - t0) * 1000)
                last_frame_id = frame_id
                have_frame = True
                if not self.paused:
                    loop_times.append(time.perf_counter())
            else:
                time.sleep(0.001)  # no new frame: don't busy-spin re-rendering the same texture

            if self.paused:
                loop_times.clear()

            compute_fps = 1000.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            achieved_fps = ((len(loop_times) - 1) / (loop_times[-1] - loop_times[0])
                            if len(loop_times) >= 2 and loop_times[-1] > loop_times[0] else 0.0)
            pos_s = index / self.fps
            dur_s = self.frame_count / self.fps
            status = "PAUSED" if self.paused else f"{achieved_fps:.0f} FPS (displayed) | {compute_fps:.0f} FPS (GPU)"
            title = f"{self.window_name}  -  {status}  -  {format_time(pos_s)} / {format_time(dur_s)}"
            self._gl_present(title, have_frame)

            now = time.perf_counter()
            if not self.paused and now - last_print >= 1.0:
                print(f"[videoplayer] {achieved_fps:5.0f} FPS (displayed) | {compute_fps:5.0f} FPS (GPU) "
                      f"| {format_time(pos_s)} / {format_time(dur_s)}")
                last_print = now

        # end of the loop
        self._reader.stop()
        self._reader.join(timeout=1.0)
        if self.audio is not None:
            self.audio.stop()
        self._surface.close()
