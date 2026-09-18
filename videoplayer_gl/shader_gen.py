"""
Compiles a SR_FastEDSR_Multi checkpoint into a list of standalone GLSL fragment-shader passes.

This mirrors the convolution-unrolling math of scripts/glsl/glsl_script_fastedsr_multi.py (which
targets mpv's `//!HOOK` shader format), but emits plain GLSL 3.30 that our own GL engine
(gl_runtime.GLUpscaler) runs as an explicit FBO render graph. Nothing here is mpv-specific.

Model recap (SR_FastEDSR_Multi):
    net.0                      Conv 3 -> nf                       (reads the LR frame)
    net.i.layers.0 / .2        Conv nf -> nf, ReLU between, per-block residual skip   (i = 1..nb)
    upscale_block_{s}.0        Conv nf -> 3*s^2                    (feature -> shuffle channels)
    PixelShuffle(s)            rearrange to HR, then add bilinear-upscaled LR frame  (base + res)

Channels are packed 4-per-texture (RGBA), so a layer with C channels becomes ceil(C/4) "groups"
(one output texture each). When a conv reads more input groups than fit in the GPU texture-unit
budget, the inputs are split into chunks and the partial sums are accumulated across chunk passes.
"""

from dataclasses import dataclass, field
from math import ceil
from typing import List, Optional, Tuple

import numpy as np

INPUT = "INPUT"  # sentinel texture key for the decoded LR frame (RGBA8, RGB in .rgb, [0,1])


@dataclass
class Pass:
    """One fragment-shader render pass. Writes texture `save` from the bound input textures."""
    save: str                                   # output texture key
    frag: str                                   # full GLSL fragment shader source
    binds: List[Tuple[str, str]]                # (sampler uniform name, source texture key)
    out_scale: int = 1                          # output res = LR res * out_scale
    fmt: str = "f16"                            # "f16" (feature) or "rgba8" (final)
    is_final: bool = False


# --------------------------------------------------------------------------------------------
# Weight/bias -> GLSL literal helpers (mirror the mpv generator; kept local for self-containment)
# --------------------------------------------------------------------------------------------
def _mat4(W, out_group, in_group, x, y) -> str:
    """4x4 weight block mapping input group -> output group for kernel tap (x, y)."""
    mat = W[4 * out_group:4 * out_group + 4, 4 * in_group:4 * in_group + 4, y, x]
    if mat.shape != (4, 4):  # pad partial groups (e.g. 3-channel input) to 4x4 with zeros
        padded = np.zeros((4, 4), dtype=mat.dtype)
        padded[:mat.shape[0], :mat.shape[1]] = mat
        mat = padded
    # GLSL mat4 is column-major, so transpose before flattening.
    return "mat4(" + ", ".join(f"{w:.8f}" for w in mat.T.flatten()) + ")"


def _bias(b, out_group) -> str:
    bias = b[4 * out_group: 4 * out_group + 4]
    if len(bias) < 4:
        bias = np.append(bias, [0.0] * (4 - len(bias)))
    return "vec4(" + ", ".join(f"{v:.8f}" for v in bias) + ")"


def _conv_lines(W, out_group, in_group) -> str:
    """The 9 mat4*vec4 taps for one (output group, input group) pair."""
    lines = ""
    for x in range(3):
        for y in range(3):
            lines += f"    result += {_mat4(W, out_group, in_group, x, y)} * get_{in_group}({x - 1},{y - 1});\n"
    return lines


# --------------------------------------------------------------------------------------------
# Shader templates
# --------------------------------------------------------------------------------------------
# Shared vertex shader: a full-screen triangle strip generated from gl_VertexID (no VBO needed).
VERT_SRC = """#version 330 core
const vec2 verts[4] = vec2[4](vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(1.0,1.0));
void main() { gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0); }
"""

# Neighbour fetch with zero-padding at borders, matching PyTorch conv padding (CLAMP would differ).
_SAMPLER_FN = """vec4 T(sampler2D t, vec2 o) {
    vec2 c = (gl_FragCoord.xy + o) * in_texel;
    if (c.x < 0.0 || c.y < 0.0 || c.x >= 1.0 || c.y >= 1.0) return vec4(0.0);
    return texture(t, c);
}
"""


def _conv_frag(W, b, out_group, in_groups: List[int],
               is_first_chunk: bool, is_last_chunk: bool,
               relu: bool, has_skip: bool) -> str:
    """Fragment shader for one output group over one chunk of input groups."""
    src = "#version 330 core\nout vec4 fragColor;\nuniform vec2 in_texel;\n"
    for i in in_groups:
        src += f"uniform sampler2D tex_g{i};\n"
    if not is_first_chunk:
        src += "uniform sampler2D tex_prev;\n"
    if is_last_chunk and has_skip:
        src += "uniform sampler2D tex_skip;\n"

    src += _SAMPLER_FN
    for i in in_groups:
        src += f"#define get_{i}(x, y) T(tex_g{i}, vec2(x, y))\n"

    src += "void main() {\n    vec4 result = vec4(0.0);\n"
    for i in in_groups:
        src += _conv_lines(W, out_group, i)
    if not is_first_chunk:
        src += "    result += texture(tex_prev, gl_FragCoord.xy * in_texel);\n"
    if is_last_chunk:
        src += f"    result += {_bias(b, out_group)};\n"
        if has_skip:
            src += "    result += texture(tex_skip, gl_FragCoord.xy * in_texel);\n"
        if relu:
            src += "    result = max(result, vec4(0.0));\n"
    src += "    fragColor = result;\n}\n"
    return src


def _conv_frag_main(W, b, out_group) -> str:
    """First conv: reads the LR frame (INPUT) as the single, zero-padded input group 0."""
    src = "#version 330 core\nout vec4 fragColor;\nuniform vec2 in_texel;\nuniform sampler2D tex_g0;\n"
    src += _SAMPLER_FN
    src += "#define get_0(x, y) T(tex_g0, vec2(x, y))\n"
    src += "void main() {\n    vec4 result = vec4(0.0);\n"
    src += _conv_lines(W, out_group, 0)
    src += f"    result += {_bias(b, out_group)};\n    fragColor = result;\n}}\n"
    return src


def _shuffle_frag(scale: int, num_groups: int) -> str:
    """PixelShuffle(scale) + bilinear base. Reads the shuffle-conv groups and the LR frame."""
    ss = scale * scale

    def pick(color: int) -> str:  # runtime idx (0..ss-1) -> the right group/element, built statically
        expr = "0.0"
        for k in range(ss):
            flat = color * ss + k
            expr = f"(idx == {k} ? p{flat // 4}[{flat % 4}] : {expr})"
        return expr

    src = "#version 330 core\nout vec4 fragColor;\n"
    src += "uniform vec2 lr_res;\nuniform vec2 hr_res;\nuniform sampler2D tex_in;\n"
    for g in range(num_groups):
        src += f"uniform sampler2D tex_g{g};\n"
    src += "void main() {\n"
    src += f"    ivec2 sub = ivec2(gl_FragCoord.xy) % {scale};\n"
    src += f"    int idx = sub.x + sub.y * {scale};\n"
    src += f"    vec2 lc = (floor(gl_FragCoord.xy / float({scale})) + 0.5) / lr_res;\n"
    for g in range(num_groups):
        src += f"    vec4 p{g} = texture(tex_g{g}, lc);\n"
    src += f"    vec3 res = vec3({pick(0)}, {pick(1)}, {pick(2)});\n"
    src += "    vec3 base = texture(tex_in, gl_FragCoord.xy / hr_res).rgb;\n"
    src += "    fragColor = vec4(clamp(base + res, 0.0, 1.0), 1.0);\n}\n"
    return src


# --------------------------------------------------------------------------------------------
# Graph builder
# --------------------------------------------------------------------------------------------
def _conv_passes(W, b, name: str, prev_key: Optional[str], out_ch: int, in_ch: int,
                 relu: bool, skip_key: Optional[str], chunk_size: int) -> Tuple[List[Pass], List[str]]:
    """Emit passes for one conv layer. Returns (passes, output texture keys, one per group)."""
    num_out = ceil(out_ch / 4)
    passes: List[Pass] = []
    out_keys: List[str] = []

    for g in range(num_out):
        save_g = f"{name}_p{g}"
        out_keys.append(save_g)

        if prev_key is None:  # first conv, reads INPUT
            passes.append(Pass(save=save_g, frag=_conv_frag_main(W, b, g),
                               binds=[("tex_g0", INPUT)]))
            continue

        num_in = ceil(in_ch / 4)
        num_chunks = ceil(num_in / chunk_size)
        for c in range(num_chunks):
            lo, hi = c * chunk_size, min((c + 1) * chunk_size, num_in)
            groups = list(range(lo, hi))
            first, last = c == 0, c == num_chunks - 1
            frag = _conv_frag(W, b, g, groups, first, last, relu, skip_key is not None)

            binds = [(f"tex_g{i}", f"{prev_key}_p{i}") for i in groups]
            if not first:
                binds.append(("tex_prev", f"{name}_p{g}_c{lo - chunk_size}"))
            if last and skip_key is not None:
                binds.append(("tex_skip", f"{skip_key}_p{g}"))

            save = save_g if last else f"{name}_p{g}_c{lo}"
            passes.append(Pass(save=save, frag=frag, binds=binds))

    return passes, out_keys


def build_passes(state_dict_np: dict, num_blocks: int, nf: int, scale: int,
                 chunk_size: int = 8) -> List[Pass]:
    """Compile the whole model into an ordered list of render passes for the chosen scale."""
    W = state_dict_np
    passes: List[Pass] = []

    # net.0 : 3 -> nf
    p, prev = _conv_passes(W["net.0.weight"], W["net.0.bias"], "net_0", None, nf, 3,
                           relu=False, skip_key=None, chunk_size=chunk_size)
    passes += p
    last_key = "net_0"

    # residual blocks
    for i in range(1, num_blocks + 1):
        l0, l2 = f"net_{i}_layers_0", f"net_{i}_layers_2"
        w0, b0 = W[f"net.{i}.layers.0.weight"], W[f"net.{i}.layers.0.bias"]
        w2, b2 = W[f"net.{i}.layers.2.weight"], W[f"net.{i}.layers.2.bias"]

        p, _ = _conv_passes(w0, b0, l0, last_key, nf, nf, relu=True, skip_key=None, chunk_size=chunk_size)
        passes += p
        p, _ = _conv_passes(w2, b2, l2, l0, nf, nf, relu=False, skip_key=last_key, chunk_size=chunk_size)
        passes += p
        last_key = l2

    # upscale conv : nf -> 3 * scale^2
    up_ch = 3 * scale * scale
    up_name = f"upscale_block_{scale}_0"
    wk, bk = W[f"upscale_block_{scale}.0.weight"], W[f"upscale_block_{scale}.0.bias"]
    p, up_keys = _conv_passes(wk, bk, up_name, last_key, up_ch, nf, relu=False, skip_key=None,
                              chunk_size=chunk_size)
    passes += p

    # PixelShuffle + bilinear base -> final HR frame
    num_groups = ceil(up_ch / 4)
    binds = [("tex_in", INPUT)] + [(f"tex_g{g}", up_keys[g]) for g in range(num_groups)]
    passes.append(Pass(save="FINAL", frag=_shuffle_frag(scale, num_groups), binds=binds,
                       out_scale=scale, fmt="rgba8", is_final=True))
    return passes
