import os
import shutil
from pathlib import Path

import numpy as np
import ncnn
import pnnx

# pnnx names the single input/output blobs in0/out0 by convention.
_IN_BLOB = "in0"
_OUT_BLOB = "out0"


def export_ncnn(onnx_path, out_dir, base_tag, input_hw):
    """Konvertuje bare-model ONNX u ncnn (.param/.bin, FP16) preko pnnx-a i kesira na disk.

    Izvor je bare ONNX (CHW RGB) -- NCNNRunner i dalje radi BGR<->RGB / normalize / clamp. ncnn je
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

    Radi pre/post obradu (BGR<->RGB, /255, HWC<->CHW, clamp) jer je konvertovan bare model bez
    VideoWrapperCV2 wrappera.
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
        # (Output has no to_pixels in this build, so the post-step stays numpy.)
        mat_in = ncnn.Mat.from_pixels(np.ascontiguousarray(bgr), ncnn.Mat.PixelType.PIXEL_BGR2RGB, w, h)
        mat_in.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0] * 3)

        ex = self.net.create_extractor()
        ex.input(_IN_BLOB, mat_in)
        _, out = ex.extract(_OUT_BLOB)

        out = np.array(out).transpose(1, 2, 0)                # (3,Hs,Ws) -> HWC RGB
        out = np.clip(out, 0.0, 1.0) * 255.0
        return np.ascontiguousarray(out[..., ::-1].astype(np.uint8))  # -> BGR uint8
