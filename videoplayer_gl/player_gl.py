"""
Pure-OpenGL SR video player.

Reuses NvdecVideoPlayer for NVDEC decode / background reader / audio / seeking, and swaps the ML
upscale step for a GLUpscaler that runs the model as GLSL shader passes. Everything stays on the
GPU: NVDEC frame -> input GL texture (CUDA interop) -> shader graph -> window, with no torch conv,
TensorRT, or ONNX runtime involved.
"""

import time
from collections import deque

import glfw

from videoplayer_nvdec.player import NvdecVideoPlayer, _DecodeReader
from .gl_runtime import GLUpscaler


class GLVideoPlayer(NvdecVideoPlayer):
    def set_engine(self, engine: GLUpscaler):
        self.engine = engine
        return self

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
                self.engine.toggle_fullscreen()
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
        if getattr(self, "engine", None) is None:
            return
        import torch

        self._print_controls()
        self._current_index = 0
        self.engine.set_key_callback(self._make_key_callback())

        self._reader = _DecodeReader(self.decoder)
        self._reader.start()
        if self.audio is not None:
            self.audio.start()
            self.audio.set_paused(False)

        frame_times = deque(maxlen=20)   # shader-graph cost per new frame
        loop_times = deque(maxlen=30)    # wall-clock of shown new frames -> achieved FPS
        last_frame_id = -1
        have_frame = False
        last_print = 0.0

        while not self.engine.should_close():
            self.engine.poll()
            frame, frame_id, index = self._reader.get_latest()
            self._current_index = index

            if frame is None:
                if not self._reader.is_alive():
                    break
                time.sleep(0.005)
                continue

            if frame_id != last_frame_id:
                t0 = time.perf_counter()
                self.engine.infer(frame)
                torch.cuda.synchronize()  # so timing reflects real GPU work, not just launch
                frame_times.append((time.perf_counter() - t0) * 1000)
                last_frame_id = frame_id
                have_frame = True
                if not self.paused:
                    loop_times.append(time.perf_counter())
            else:
                time.sleep(0.001)

            if self.paused:
                loop_times.clear()

            compute_fps = 1000.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            achieved_fps = ((len(loop_times) - 1) / (loop_times[-1] - loop_times[0])
                            if len(loop_times) >= 2 and loop_times[-1] > loop_times[0] else 0.0)
            pos_s = index / self.fps
            dur_s = self.frame_count / self.fps
            status = "PAUSED" if self.paused else f"{achieved_fps:.1f} FPS (real) | {compute_fps:.1f} FPS (compute)"

            if have_frame:
                self.engine.present(f"{self.window_name}  -  {status}  -  {int(pos_s)}s / {int(dur_s)}s")

            now = time.perf_counter()
            if not self.paused and now - last_print >= 1.0:
                print(f"[videoplayer_gl] {achieved_fps:5.1f} FPS (real) | {compute_fps:5.1f} FPS (compute) "
                      f"| {int(pos_s)}s / {int(dur_s)}s")
                last_print = now

        self._reader.stop()
        self._reader.join(timeout=1.0)
        if self.audio is not None:
            self.audio.stop()
        self.engine.close()
