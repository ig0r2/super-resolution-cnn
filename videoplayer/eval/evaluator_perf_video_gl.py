"""
Speed evaluation for the pure-OpenGL SR pipeline (players/gl.py), runtype "opengl".

Same four-stage breakdown and reporting as EvaluatorPerfVideoNVDEC (one 'total' loop that times
each part in-line, average milliseconds per frame), so its CSV row lines up column-for-column with
the TensorRT rows in results_{N}x_video_ms.csv:

  - decode  : NVDEC decode alone (identical to the TRT evaluators)
  - sr      : the SR compute alone on the decoded frame = CUDA->GL input upload + the shader
              graph. Unlike TRT (which consumes the CUDA frame directly), the input upload is an
              unavoidable part of running the shaders, so it is counted here.
  - display : the display draw alone (see 'total').
  - total   : decode + sr + the display draw. There is no device->host copy or bicubic downscale:
             the SR output already lives in a GL texture on the display surface, so the GL analog
             of the TRT "downscale + deliver" stage is the letterboxed draw into the framebuffer
             (buffer swap / vsync excluded, matching the TRT evaluators which never present).

Timing uses glFinish(), not just torch.cuda.synchronize(): the shader passes execute on the GL
command queue rather than a CUDA stream. The CUDA input upload is ordered before the passes by
cudaGraphicsUnmapResources, so glFinish() (plus a cuda sync) captures the whole chain.

Only SR_FastEDSR_Multi is supported; other architectures raise RuntimeError (the driver records a
blank row, as it does for a failed TRT engine build).
"""

import gc
import time

import torch
from OpenGL import GL

from videoplayer.scaling import choose_auto_scale
from videoplayer.backends.gl_build import build_engine
from videoplayer.decode.decoder import NvDecoder
from ._base import _BaseVideoPerfEvaluator


class EvaluatorPerfVideoGL(_BaseVideoPerfEvaluator):
    def __init__(self, checkpoint_path, name, video_path, upscale_factor=2,
                 warmup_runs=20, iterations=200, chunk_size=8):
        super().__init__(checkpoint_path, name, video_path, upscale_factor, warmup_runs, iterations)
        self.chunk_size = chunk_size

    def evaluate(self):
        torch.cuda.empty_cache()

        decoder = NvDecoder(str(self.video_path))
        h, w = decoder.height, decoder.width
        n = len(decoder)
        print(f"Video {self.video_path.name}: {w}x{h}, {n} frames, {self.upscale_factor}x")

        # Display target the player would show on a 1080p screen -> the hidden window / draw size.
        target_hw = choose_auto_scale((h, w), self._ref_screen, (self.upscale_factor,)).target_size

        engine, meta = build_engine(self.checkpoint_path, self.upscale_factor, h, w,
                                    win_w=target_hw[1], win_h=target_hw[0],
                                    visible=False, chunk_size=self.chunk_size)
        print(f"GL graph: {meta['num_passes']} passes (num_blocks={meta['num_blocks']}, nf={meta['nf']})")

        def total_step(i):
            engine.infer(decoder.frame(i % n))
            engine.draw_final()

        try:
            # Warm up the whole pipeline, then time each part inside one 'total' loop. decode is a
            # CUDA stage (cuda sync); sr/display run on the GL queue, so glFinish bounds them.
            for i in range(self.warmup_runs):
                total_step(i)
            self._sync(gl=True)

            totals = {"decode": 0.0, "sr": 0.0, "display": 0.0}
            for i in range(self.iterations):
                t0 = time.perf_counter()
                frame = decoder.frame(i % n)
                self._sync(gl=False)          # finishes NVDEC decode
                t1 = time.perf_counter()
                engine.infer(frame)
                self._sync(gl=True)           # finishes CUDA->GL upload + the shader passes
                t2 = time.perf_counter()
                engine.draw_final()
                self._sync(gl=True)           # finishes the display draw on the GL queue
                t3 = time.perf_counter()
                totals["decode"] += t1 - t0
                totals["sr"] += t2 - t1
                totals["display"] += t3 - t2

            return self._summarize(totals)
        finally:
            engine.close()
            del decoder
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def _sync(gl: bool):
        torch.cuda.synchronize()  # finishes NVDEC decode + the CUDA input upload
        if gl:
            GL.glFinish()          # finishes the shader passes / display draw on the GL queue
