"""
Video player koji koristi INT8 TensorRT engine (umesto FP16 torch-tensorrt .pt2).

Modifikacija scripts/videoplayer/video_player_trt.py:
  - Umesto torch_tensorrt FP16 kompilacije, gradi INT8 TRT engine preko cistog
    TensorRT Python API-ja (isti pristup kao scripts/evaluation/int8_trt_script.py),
    jer je modelopt INT8 na Windows-u pokvaren.
  - INT8 skale se kalibrisu na PRAVIM frejmovima iz zadatog videa (domenski
    tacna kalibracija), pa se engine kesira na disk.
  - Ulaz/izlaz su isti kao kod originala (VideoWrapperCV2): frejm (H,W,3) BGR
    uint8 -> (H*r, W*r, 3) BGR uint8. Oblik je fiksan (video ima stalnu
    rezoluciju), pa je engine statican - bez dinamickog profila.

Napomene:
  - INT8 se isplati na vecim modelima; na sitnim moze biti sporiji od FP16
    (vidi quantization-findings). Ovaj player je namenjen bas vecim modelima.
  - Za headless proveru bez GUI-ja: postavi env SR_SMOKE=1 (izmeri N frejmova
    i izadje, bez cv2 prozora).

Pokretanje:
    python scripts/videoplayer/video_player_trt_int8.py
"""

import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch
import tensorrt as trt

from utils.checkpoints import load_model_from_checkpoint
from utils.path import get_project_root, get_checkpoints_path
from utils.video.evaluator_perf_video import VideoWrapperCV2
from utils.video.model_utils import TileProcessorTorch
from utils.video.videoplayer import VideoPlayer

# ============================ Podesavanja ============================
MODEL = "multiscale/SR_FastEDSR_4_256"
VIDEO_PATH = get_project_root("videoinput/F1Bahr-480p50.mp4")
UPSCALE_FACTOR = 2
TILED = False
TILE_SIZE = 256

# "fp16" = raw-TRT FP16 (ostro, puni kvalitet; vec mnogo brze od torch_tensorrt .pt2)
# "int8" = jos brze (~1.15x preko FP16), ali moguca vidljiva mutnoca kod SR-a
PRECISION = "int8"

CALIB_FRAMES = 50          # broj frejmova za INT8 kalibraciju
WORKSPACE_GB = 4           # gornja granica TRT tactic-workspace pool-a
BUILDER_OPT_LEVEL = 2      # 1-5; nizi = manje VRAM/brzi build, malo manje optimalan engine
REBUILD_ENGINE = False     # True = ignorisi kesirani .engine i buildaj iznova

# Mesovita preciznost: koliko conv slojeva ostaviti u FP16 (nose detalj/rezidual).
# SR izlaz = bilinear baza + rezidual; INT8 zaokruzi sitan detalj -> mutnije.
# Zadrzavanje prvog (ekstrakcija) i poslednjeg (rekonstrukcija) conv-a u FP16 vraca ostrinu.
INT8_KEEP_FP16_FIRST = int(os.environ.get("SR_KEEP_FIRST", "1"))   # 0 = pun INT8
INT8_KEEP_FP16_LAST = int(os.environ.get("SR_KEEP_LAST", "1"))
SMOKE_TEST = os.environ.get("SR_SMOKE") == "1"  # headless provera bez GUI-ja
# ====================================================================

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.UINT8: torch.uint8,
}


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """Puni TRT kalibrator pravim frejmovima (fp16, oblik = ulaz mreze)."""

    def __init__(self, calib_tensors, cache_path):
        super().__init__()
        self.tensors = calib_tensors
        self.cache_path = Path(cache_path)
        self.idx = 0

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.tensors):
            return None
        t = self.tensors[self.idx]
        self.idx += 1
        return [int(t.data_ptr())]

    def read_calibration_cache(self):
        return self.cache_path.read_bytes() if self.cache_path.exists() else None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)


def export_wrapper_onnx(model, onnx_path, input_hw):
    """Eksportuje VideoWrapperCV2(model) u ONNX sa statickim ulazom (H,W,3), fp16."""
    wrapper = VideoWrapperCV2(model).eval().cuda().half()
    dummy = torch.randn(input_hw[0], input_hw[1], 3, device="cuda", dtype=torch.float16)
    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"], opset_version=17, dynamo=False,
    )


def _keep_convs_fp16(network, config, keep_first, keep_last):
    """Zadrzava prvih keep_first i poslednjih keep_last conv slojeva u FP16 (mesovita preciznost)."""
    if not (keep_first or keep_last):
        return
    conv_idx = [i for i in range(network.num_layers)
                if network.get_layer(i).type == trt.LayerType.CONVOLUTION]
    keep = set(conv_idx[:keep_first])
    if keep_last:
        keep |= set(conv_idx[len(conv_idx) - keep_last:])
    if not keep:
        return
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    for i in keep:
        layer = network.get_layer(i)
        layer.precision = trt.DataType.HALF
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.DataType.HALF)
    print(f"Mesovita preciznost: {len(keep)}/{len(conv_idx)} conv slojeva u FP16")


def build_engine(onnx_path, engine_path, precision, calibrator=None,
                 keep_first=0, keep_last=0, workspace_gb=4, opt_level=2, rebuild=False):
    """
    Gradi (ili ucitava iz kesa) staticni TRT engine.
    precision: "fp16" | "int8"  (int8 podrazumeva FP16 fallback + mesovitu preciznost)
    """
    runtime = trt.Runtime(TRT_LOGGER)
    if engine_path.exists() and not rebuild:
        print(f"Engine iz kesa: {engine_path.name}")
        return runtime.deserialize_cuda_engine(engine_path.read_bytes())

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.builder_optimization_level = opt_level  # nizi -> manje VRAM/brzi build
    config.set_flag(trt.BuilderFlag.FP16)
    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        _keep_convs_fp16(network, config, keep_first, keep_last)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed (build_serialized_network vratio None)")

    engine_path.write_bytes(bytes(serialized))
    engine = runtime.deserialize_cuda_engine(serialized)

    del builder, network, parser, config, serialized
    gc.collect()
    torch.cuda.empty_cache()
    return engine


class TRTVideoRunner:
    """Pokrece staticni TRT engine nad frejmom (H,W,3) i vraca uint8 (H*r,W*r,3)."""

    def __init__(self, engine):
        self.engine = engine
        self.context = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_dtype = _TRT_TO_TORCH[engine.get_tensor_dtype(self.in_name)]
        self.out_dtype = _TRT_TO_TORCH[engine.get_tensor_dtype(self.out_name)]

    def __call__(self, x):
        x = x.to(self.in_dtype).contiguous()
        self.context.set_input_shape(self.in_name, tuple(x.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        out = torch.empty(out_shape, dtype=self.out_dtype, device="cuda")
        self.context.set_tensor_address(self.in_name, x.data_ptr())
        self.context.set_tensor_address(self.out_name, out.data_ptr())
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        if out.dtype != torch.uint8:  # ako TRT ne izlazi uint8, konvertuj ovde
            out = out.clamp(0, 255).to(torch.uint8)
        return out


def get_calib_frames(video_path, input_hw, tiled, n):
    """Cita n frejmova ravnomerno iz videa -> lista fp16 GPU tenzora (H,W,3) BGR [0-255]."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    step = max(1, total // n)
    ih, iw = input_hw
    frames = []
    i = 0
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            if tiled:  # centralni (TILE,TILE) isecak
                h, w, _ = frame.shape
                top = max(0, (h - ih) // 2)
                left = max(0, (w - iw) // 2)
                frame = frame[top:top + ih, left:left + iw, :]
            frames.append(torch.from_numpy(np.ascontiguousarray(frame)).cuda().half())
        i += 1
    cap.release()
    return frames


# ============================ Main ============================
player = VideoPlayer(VIDEO_PATH)
INPUT_SIZE = (TILE_SIZE, TILE_SIZE) if TILED else player.size  # (H, W)

checkpoint_name = MODEL.split("/", 1)[-1]
build_dir = get_project_root("exports/trt_int8")
build_dir.mkdir(parents=True, exist_ok=True)
base_tag = f"{checkpoint_name}_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_{UPSCALE_FACTOR}x_cv2"
onnx_path = build_dir / f"{base_tag}.onnx"
if PRECISION == "int8":
    eng_tag = f"{base_tag}_int8_k{INT8_KEEP_FP16_FIRST}-{INT8_KEEP_FP16_LAST}"
else:
    eng_tag = f"{base_tag}_fp16"
engine_path = build_dir / f"{eng_tag}.engine"


def ensure_onnx():
    """Eksportuje ONNX ako ne postoji (potreban za bilo koji build)."""
    if onnx_path.exists() and not REBUILD_ENGINE:
        return
    checkpoint_path = get_checkpoints_path(f"{MODEL}.pth")
    if not checkpoint_path.exists():
        print(f"Checkpoint za {MODEL} ne postoji")
        sys.exit(1)
    model, _ = load_model_from_checkpoint(checkpoint_path, "cpu")
    model.upscale_factor = UPSCALE_FACTOR
    export_wrapper_onnx(model, onnx_path, INPUT_SIZE)
    print(f"ONNX -> {onnx_path.name}")


# Build (ili ucitaj iz kesa) engine za izabranu preciznost
if REBUILD_ENGINE or not engine_path.exists():
    print(f"{PRECISION.upper()} engine za {INPUT_SIZE[0]}x{INPUT_SIZE[1]} - build...")
    ensure_onnx()
    if PRECISION == "int8":
        calib_frames = get_calib_frames(VIDEO_PATH, INPUT_SIZE, TILED, CALIB_FRAMES)
        print(f"Kalibracija: {len(calib_frames)} frejmova @ {INPUT_SIZE}")
        calib = Int8Calibrator(calib_frames, build_dir / f"{eng_tag}.cache")
        engine = build_engine(onnx_path, engine_path, "int8", calib,
                              INT8_KEEP_FP16_FIRST, INT8_KEEP_FP16_LAST,
                              WORKSPACE_GB, BUILDER_OPT_LEVEL, REBUILD_ENGINE)
        del calib_frames
    else:
        engine = build_engine(onnx_path, engine_path, "fp16",
                              workspace_gb=WORKSPACE_GB, opt_level=BUILDER_OPT_LEVEL, rebuild=REBUILD_ENGINE)
    torch.cuda.empty_cache()
else:
    engine = build_engine(None, engine_path, PRECISION)

runner = TRTVideoRunner(engine)
print(f"Engine ({PRECISION}) I/O dtype: in={runner.in_dtype}, out={runner.out_dtype}")


def infer(tile):
    return runner(tile)


if TILED:
    tile_processor = TileProcessorTorch(upscale_factor=UPSCALE_FACTOR, tile_size=TILE_SIZE, overlap=8)

    def upscale(frame):
        frame_gpu = torch.from_numpy(frame).cuda()
        return tile_processor.process_frame(frame_gpu, infer).cpu().numpy()
else:
    def upscale(frame):
        frame_gpu = torch.from_numpy(frame).cuda()
        return infer(frame_gpu).cpu().numpy()


if SMOKE_TEST:
    # Headless: izmeri brzinu INT8 vs FP16 i kvalitet (PSNR INT8 vs FP16 referenca)
    import math
    import time

    # FP16 referentni engine (za poredjenje kvaliteta)
    fp16_engine_path = build_dir / f"{base_tag}_fp16.engine"
    if REBUILD_ENGINE or not fp16_engine_path.exists():
        ensure_onnx()
        fp16_engine = build_engine(onnx_path, fp16_engine_path, "fp16",
                                   workspace_gb=WORKSPACE_GB, opt_level=BUILDER_OPT_LEVEL, rebuild=REBUILD_ENGINE)
    else:
        fp16_engine = build_engine(None, fp16_engine_path, "fp16")
    fp16_runner = TRTVideoRunner(fp16_engine)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    N = 60
    frames = []
    while len(frames) < N:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    print(f"Frejm: {frames[0].shape} {frames[0].dtype} -> izlaz {upscale(frames[0]).shape}")

    g0 = torch.from_numpy(frames[0]).cuda()
    for _ in range(10):
        runner(g0); fp16_runner(g0)
    torch.cuda.synchronize()

    t8 = t16 = mse_sum = 0.0
    for fr in frames:
        g = torch.from_numpy(fr).cuda()
        torch.cuda.synchronize(); a = time.time(); o8 = runner(g); torch.cuda.synchronize(); t8 += time.time() - a
        torch.cuda.synchronize(); a = time.time(); o16 = fp16_runner(g); torch.cuda.synchronize(); t16 += time.time() - a
        mse_sum += torch.mean((o8.float() - o16.float()) ** 2).item()
    n = len(frames)
    psnr = 10 * math.log10(255 * 255 / (mse_sum / n)) if mse_sum > 0 else 99.0
    print("-" * 40)
    print(f"Frejmova: {n}  | keep_fp16 first/last = {INT8_KEEP_FP16_FIRST}/{INT8_KEEP_FP16_LAST}")
    print(f"INT8: {n / t8:.1f} FPS (samo engine)")
    print(f"FP16: {n / t16:.1f} FPS (samo engine)")
    print(f"PSNR INT8 vs FP16: {psnr:.2f} dB  (vise = blize FP16 = manje mutno)")
else:
    player.set_upscale_fn(upscale).play()
