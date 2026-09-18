"""
Pure-OpenGL super-resolution player.

This package upscales video frames with **no ML runtime** (no TensorRT, ONNX, or torch conv):
the FastEDSR model is compiled into a chain of GLSL fragment-shader passes that run entirely on
the GPU, the same idea as the mpv `--glsl-shader` export in scripts/glsl/, but driven by our own
minimal render-graph engine and fed from NVDEC-decoded frames via CUDA<->GL interop.

It is a self-contained sibling of `videoplayer_nvdec` and does not modify it. See run_gl.py for
the entry point and validate.py for a numeric parity check against the PyTorch model.
"""
