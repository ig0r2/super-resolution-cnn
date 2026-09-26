import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import ncnn
import pnnx
import torch

# pnnx names the single input/output blobs in0/out0 by convention.
_IN_BLOB = "in0"
_OUT_BLOB = "out0"


class _NCNNOutput(torch.nn.Module):
    """Model + x*255 i clamp(0,255) na izlazu, da ncnn vrati CHW RGB spreman za float->uint8 na CPU.
    BGR i HWC namerno nisu u grafu (Slice/Concat/Permute na ncnn GPU kostaju vise nego sto stede)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return (self.model(x) * 255.0).clamp(0.0, 255.0)


def export_onnx_ncnn(model, path, input_hw):
    """Eksportuje model (CHW RGB [0,1] ulaz, CHW RGB [0,255] izlaz) u fp32 onnx.
    Koristi se za ncnn (pnnx). Mora fp32 jer pnnx ne pravi fp16 slojeve ispravno.
    pnnx svakako sam pretvori iz fp32 u fp16."""
    model = _NCNNOutput(model).eval().cpu().float()
    dummy = torch.rand(1, 3, input_hw[0], input_hw[1])
    torch.onnx.export(model, dummy, str(path), input_names=["input"],
                      output_names=["output"], opset_version=17, dynamo=False)


def export_ncnn(onnx_path, out_dir, base_tag, input_hw):
    """Konvertuje ncnn-source ONNX u ncnn (.param/.bin, FP16) preko pnnx-a i kesira na disk.

    Izvor je ONNX iz export_onnx_ncnn (CHW RGB [0,1] -> CHW RGB [0,255], clamp u grafu) -- NCNNRunner
    i dalje radi BGR<->RGB / normalize / float->uint8. ncnn je
    size-agnostic (fully-conv + relative Interp scale), pa je input_hw samo primer oblika koji pnnx
    trazi za onnx ulaz; keš je ionako po rezoluciji.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    param_path = out_dir / f"{base_tag}.ncnn.param"
    bin_path = out_dir / f"{base_tag}.ncnn.bin"

    # pnnx.convert re-imports a generated _pnnx.py that embeds file paths; an absolute C:\Users\...
    # path trips a '\U' unicodeescape crash, so run from inside out_dir with a LOCAL relative copy of
    # the ONNX and relative output names (same trick the old bare-model export used).
    local_onnx = out_dir / f"{base_tag}.onnx"
    shutil.copyfile(onnx_path, local_onnx)
    prev_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        # input_types MORA "f32" (usaglaseno sa fp32 ONNX-om iz export_onnx_ncnn): sa "f16" pnnx ne spusti Conv u
        # ncnn i model puca pri ucitavanju. fp16=True i dalje daje fp16 ncnn tezine.
        pnnx.convert(f"{base_tag}.onnx",
                     input_shapes=[[1, 3, input_hw[0], input_hw[1]]], input_types=["f32"],
                     ncnnparam=f"{base_tag}.ncnn.param", ncnnbin=f"{base_tag}.ncnn.bin", fp16=True)
    finally:
        os.chdir(prev_cwd)

    # pnnx also dumps the local .onnx copy + .pnnx.* / *_pnnx.py / *_ncnn.py intermediates next to the
    # output; keep only the ncnn engine files (the persistent ONNX cache lives elsewhere).
    for leftover in out_dir.glob(f"{base_tag}*"):
        if leftover.is_file() and leftover not in (param_path, bin_path):
            leftover.unlink()

    return param_path, bin_path


class NCNNRunner:
    """Pokrece ncnn-Vulkan engine nad BGR uint8 frejmom (H,W,3) i vraca BGR uint8 (H*s,W*s,3).

    Model vec radi *255 i clamp (export_onnx_ncnn); ovde ostaje BGR<->RGB, /255, HWC<->CHW i
    float->uint8, jer VideoWrapperCV2 wrapper nije u ncnn grafu.
    """

    def __init__(self, param_path, bin_path, use_vulkan=True):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = use_vulkan
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = True
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        # from_pixels folds BGR->RGB + HWC->CHW into one ncnn call; normalize does the /255.
        # (Output has no to_pixels in this build, so the post-step is done with cv2.)
        mat_in = ncnn.Mat.from_pixels(np.ascontiguousarray(bgr), ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0] * 3)

        ex = self.net.create_extractor()
        ex.input(_IN_BLOB, mat_in)
        _, out = ex.extract(_OUT_BLOB)

        # CHW RGB float view (no copy), already *255 and clamped by the model; convertScaleAbs is a plain
        # rounding float->uint8 per plane, merge in reversed order gives HWC BGR.
        chw = np.asarray(out)
        return cv2.merge([cv2.convertScaleAbs(chw[c]) for c in (2, 1, 0)])
