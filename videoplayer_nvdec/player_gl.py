"""
CUDA<->OpenGL variant of the NVDEC player: the SR output is shown straight from GPU memory via a
CUDA-registered GL texture (see gl_display.GLDisplay), with no device->host copy and no cv2.

Reuses NvdecVideoPlayer for decode/reader/audio/seek and only replaces the display loop. Frame
stats (FPS, position) go to the window title bar instead of being drawn onto the frame, which
avoids a CPU text-draw pass and any GL text rendering.
"""

import time
from collections import deque
from typing import Optional

import glfw
import torch
import torch.nn.functional as F

from .gl_display import GLDisplay
from .player import NvdecVideoPlayer, _DecodeReader


class NvdecGLVideoPlayer(NvdecVideoPlayer):
    def _downscale(self, out_chw: torch.Tensor) -> torch.Tensor:
        """(3,H*s,W*s) uint8 RGB CUDA -> (3,Ht,Wt) uint8 RGB CUDA, GPU bicubic if a target is set."""
        if self.target_size is None:
            return out_chw
        x = out_chw.unsqueeze(0).float()
        x = F.interpolate(x, size=self.target_size, mode="bicubic", align_corners=False)
        return x.clamp(0.0, 255.0).to(torch.uint8).squeeze(0)

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
            elif key in (glfw.KEY_F, glfw.KEY_M):
                self._display.toggle_fullscreen()
        return on_key

    def _print_controls(self):
        print("Controls:")
        print("  Space       Pause / play")
        print(f"  Left/Right  Seek -{self.seek_step_s:g}s / +{self.seek_step_s:g}s")
        print(f"  Up/Down     Seek +{self.seek_step_large_s:g}s / -{self.seek_step_large_s:g}s")
        print("  f / m       Toggle fullscreen")
        print("  q / Esc     Quit")
        if self.audio is not None and not self.audio.available:
            print("Audio: no track / ffmpeg / sounddevice available, running silent.")

    def play(self):
        if self.upscale_fn is None:
            return

        self._print_controls()
        self._current_index = 0

        # Window starts at the display target if we have one, else the native frame size.
        win_h, win_w = self.target_size if self.target_size is not None else self.frame_size
        self._display = GLDisplay(win_w, win_h, window_name_or_default(self.window_name),
                                  fullscreen=self.fullscreen)
        glfw.set_key_callback(self._display.window, self._make_key_callback())

        self._reader = _DecodeReader(self.decoder)
        self._reader.start()
        if self.audio is not None:
            self.audio.start()
            self.audio.set_paused(False)

        frame_times = deque(maxlen=20)   # SR + downscale + upload cost per new frame
        loop_times = deque(maxlen=30)    # wall-clock of shown new frames -> achieved FPS
        last_frame_id = -1
        have_frame = False
        last_print = 0.0                 # throttle console FPS to ~1 Hz (visible in fullscreen)

        while not self._display.should_close():
            self._display.poll()
            frame, frame_id, index = self._reader.get_latest()
            self._current_index = index

            if frame is None:
                if not self._reader.is_alive():
                    break
                time.sleep(0.005)
                continue

            if frame_id != last_frame_id:
                t0 = time.perf_counter()
                out = self.upscale_fn(frame)
                self._display.upload(self._downscale(out))
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

            if have_frame:
                self._display.render()

            compute_fps = 1000.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            achieved_fps = ((len(loop_times) - 1) / (loop_times[-1] - loop_times[0])
                            if len(loop_times) >= 2 and loop_times[-1] > loop_times[0] else 0.0)
            pos_s = index / self.fps
            dur_s = self.frame_count / self.fps
            status = "PAUSED" if self.paused else f"{achieved_fps:.1f} FPS (real) | {compute_fps:.1f} FPS (compute)"
            self._display.set_title(f"{self.window_name}  -  {status}  -  {int(pos_s)}s / {int(dur_s)}s")

            now = time.perf_counter()
            if not self.paused and now - last_print >= 1.0:
                print(f"[videoplayer_nvdec] {achieved_fps:5.1f} FPS (real) | {compute_fps:5.1f} FPS (compute) "
                      f"| {int(pos_s)}s / {int(dur_s)}s")
                last_print = now

        self._reader.stop()
        self._reader.join(timeout=1.0)
        if self.audio is not None:
            self.audio.stop()
        self._display.close()


def window_name_or_default(name: Optional[str]) -> str:
    return name if name else "SR Video (GPU)"
