"""
Speed evaluation for the pure-OpenGL SR pipeline (videoplayer_gl), runtype "opengl".

Same four-stage breakdown and reporting as EvaluatorPerfVideoNVDEC, so its CSV row lines up
column-for-column with the TensorRT rows in results_{N}x_video_FPS.csv:

  - decode : NVDEC decode alone (identical to the TRT evaluators)
  - sr     : the SR compute alone on a GPU-resident frame = CUDA->GL input upload + the shader
             graph. Unlike TRT (which consumes the CUDA frame directly), the input upload is an
             unavoidable part of running the shaders, so it is counted here.
  - e2e    : decode + sr back to back (what the player's upscale step sustains)
  - full   : decode + sr + the display draw. There is no device->host copy or bicubic downscale:
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
from pathlib import Path

import torch
from OpenGL import GL

from videoplayer.scaling import choose_auto_scale
from .build import build_engine
from videoplayer_nvdec.decoder import NvDecoder

# Reference screen for the display downscale target, so "full" is monitor-independent (matches
# EvaluatorPerfVideoNVDEC).
_REF_SCREEN = (1080, 1920)


class EvaluatorPerfVideoGL:
    def __init__(self, checkpoint_path, name, video_path, upscale_factor=2,
                 warmup_runs=20, iterations=200, chunk_size=8):
        self.checkpoint_path = Path(checkpoint_path)
        self.name = name
        self.upscale_factor = upscale_factor
        self.video_path = Path(video_path)
        self.warmup_runs = warmup_runs
        self.iterations = iterations
        self.chunk_size = chunk_size

    def evaluate(self):
        torch.cuda.empty_cache()

        decoder = NvDecoder(str(self.video_path))
        h, w = decoder.height, decoder.width
        n = len(decoder)
        print(f"Video {self.video_path.name}: {w}x{h}, {n} frames, {self.upscale_factor}x")

        # Display target the player would show on a 1080p screen -> the hidden window / draw size.
        target_hw = choose_auto_scale((h, w), _REF_SCREEN, (self.upscale_factor,)).target_size

        engine, meta = build_engine(self.checkpoint_path, self.upscale_factor, h, w,
                                    win_w=target_hw[1], win_h=target_hw[0],
                                    visible=False, chunk_size=self.chunk_size)
        print(f"GL graph: {meta['num_passes']} passes (num_blocks={meta['num_blocks']}, nf={meta['nf']})")

        def decode(i):
            return decoder.frame(i % n)

        def sr(_i):
            engine.infer(fixed)

        def e2e(i):
            engine.infer(decoder.frame(i % n))

        def full(i):
            engine.infer(decoder.frame(i % n))
            engine.draw_final()

        try:
            fixed = decoder.frame(0)  # GPU-resident frame for the SR-only measurement
            decode_fps = self._time("decode", decode, gl=False)
            sr_fps = self._time("sr", sr, gl=True)
            e2e_fps = self._time("e2e (decode+SR)", e2e, gl=True)
            full_fps = self._time("full (decode+SR+display)", full, gl=True)

            print("-" * 30)
            print(f"decode : {decode_fps}")
            print(f"sr     : {sr_fps}")
            print(f"e2e    : {e2e_fps}")
            print(f"full   : {full_fps}")
            print("-" * 30)

            return {"decode": decode_fps, "sr": sr_fps, "e2e": e2e_fps, "full": full_fps}
        finally:
            engine.close()
            del decoder
            gc.collect()
            torch.cuda.empty_cache()

    def _time(self, label, fn, gl: bool):
        for i in range(self.warmup_runs):
            fn(i)
        self._sync(gl)

        start = time.perf_counter()
        for i in range(self.iterations):
            fn(i)
        self._sync(gl)
        total = time.perf_counter() - start

        avg_ms = (total / self.iterations) * 1000
        fps = 1.0 / (avg_ms / 1000)
        print(f"  {label:28s} {avg_ms:7.2f} ms/frame  ->  {fps:8.1f} FPS")
        return f"{fps:.2f}"

    @staticmethod
    def _sync(gl: bool):
        torch.cuda.synchronize()  # finishes NVDEC decode + the CUDA input upload
        if gl:
            GL.glFinish()          # finishes the shader passes / display draw on the GL queue
