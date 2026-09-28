"""
Build the player for a (runtime, decoder, display) choice: the one place that knows which
combinations exist. Only the chosen player module is imported, so e.g. the CPU-decode path never
pulls in NVDEC / glfw, and the opengl path never pulls in TensorRT.

    runtime                                         decoder            display  player
    tensorrt, tensorrt-pt2, ncnn-vulkan, onnxruntime-*  pyav, cv2          cv2      VideoPlayerCV2
    tensorrt                                        nvdec              cv2      VideoPlayerNvdecCV2
    tensorrt                                        nvdec              opengl   VideoPlayerNvdecGL
    opengl                                          nvdec, pyav, cv2   opengl   VideoPlayerGL
"""

from typing import Literal, get_args

from ..backends.cv2_backends import Runtype
from ..decode import Decoder

Runtime = Literal[Runtype, "opengl"]
Display = Literal["cv2", "opengl"]

_COMBINATIONS = [
    ("runtime", "decoder", "display", "player"),
    ("tensorrt, tensorrt-pt2, ncnn-vulkan, onnxruntime-*", "pyav, cv2", "cv2", "VideoPlayerCV2"),
    ("tensorrt", "nvdec", "cv2", "VideoPlayerNvdecCV2"),
    ("tensorrt", "nvdec", "opengl", "VideoPlayerNvdecGL"),
    ("opengl", "nvdec, pyav, cv2", "opengl", "VideoPlayerGL"),
]


def _table(rows) -> str:
    """ASCII grid of rows (first row = header), plain ASCII so any console can print it."""
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([sep, line(rows[0]), sep.replace("-", "=")] + [line(r) for r in rows[1:]] + [sep])


SUPPORTED = "Supported combinations:\n" + _table(_COMBINATIONS)


def check_combination(runtime: Runtime, decoder: Decoder, display: Display):
    """Raise ValueError (with the supported combinations) if the player can't run this choice."""
    for name, value, choices in (("runtime", runtime, get_args(Runtime)),
                                 ("decoder", decoder, get_args(Decoder)),
                                 ("display", display, get_args(Display))):
        if value not in choices:
            raise ValueError(f"Unknown {name} {value!r} (expected one of: {', '.join(choices)})")

    if runtime == "opengl":
        reason = None if display == "opengl" else "the opengl runtime renders into its own OpenGL window (display 'opengl')"
    elif decoder == "nvdec":
        reason = None if runtime == "tensorrt" else "NVDEC frames are CUDA tensors, which only the 'tensorrt' runtime takes"
    else:
        reason = None if display == "cv2" else "the opengl display takes GPU frames: use decoder 'nvdec' (runtime 'tensorrt') or runtime 'opengl'"

    if reason is not None:
        raise ValueError(f"Unsupported combination runtime={runtime!r}, decoder={decoder!r}, "
                         f"display={display!r}: {reason}.\n{SUPPORTED}")


def make_player(video_path, model, runtime: Runtime, decoder: Decoder, display: Display,
                candidate_scales=(2, 3, 4), **kwargs):
    """Validate the combination, then build the matching player (which opens the video, picks the
    scale and builds its SR backend). chunk_size is used by the opengl runtime only; kwargs
    (target_size, enable_audio, start_fullscreen) go to the player."""
    check_combination(runtime, decoder, display)

    if runtime == "opengl":
        from .player_gl import VideoPlayerGL
        return VideoPlayerGL(video_path, model, decoder=decoder, candidate_scales=candidate_scales, **kwargs)
    if decoder == "nvdec":
        if display == "opengl":
            from .player_nvdec_gl import VideoPlayerNvdecGL as Player
        else:
            from .player_nvdec import VideoPlayerNvdecCV2 as Player
        return Player(video_path, model, candidate_scales=candidate_scales, **kwargs)

    from .player_cv2 import VideoPlayerCV2
    return VideoPlayerCV2(video_path, model, runtype=runtime, decoder=decoder,
                          candidate_scales=candidate_scales, **kwargs)
