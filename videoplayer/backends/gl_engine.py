"""
Minimal OpenGL render-graph engine that executes the shader passes from shader_gen.

Per frame:
  1. the NVDEC frame (a (3,H,W) uint8 RGB CUDA tensor) is copied device->device into an
     OpenGL input texture via CUDA<->GL interop (no host round-trip, same trick as
     backends/gl_display.py);
  2. each Pass renders into its own offscreen texture (RGBA16F for features, RGBA8 for the
     final HR frame) at the model's exact resolution;
  3. the final HR texture is drawn to the window, letterboxed, or read back for validation.

Orientation convention (kept consistent end to end): image row 0 (top) is uploaded to texture
row 0, which GL samples at v=0 and the display shader draws at the top of the window.
"""

from typing import Dict, List, Optional, Tuple

import glfw
import numpy as np
import torch
from OpenGL import GL
from cuda.bindings import runtime as rt

from .shader_gen import INPUT, VERT_SRC, Pass

_DISPLAY_VERT = """#version 330 core
const vec2 verts[4] = vec2[4](vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(-1.0,1.0), vec2(1.0,1.0));
out vec2 uv;
void main() {
    // Image top is stored at texture v=0, so flip v: window top (y=+1) samples v=0.
    uv = vec2((verts[gl_VertexID].x + 1.0) * 0.5, 1.0 - (verts[gl_VertexID].y + 1.0) * 0.5);
    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}
"""
_DISPLAY_FRAG = """#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D tex;
void main() { color = texture(tex, uv); }
"""


def _cuda_check(err, msg: str = ""):
    if isinstance(err, tuple):
        err = err[0]
    if err != rt.cudaError_t.cudaSuccess:
        _, name = rt.cudaGetErrorName(err)
        _, desc = rt.cudaGetErrorString(err)
        raise RuntimeError(f"CUDA error in {msg}: {name.decode()} - {desc.decode()}")


class GLUpscaler:
    def __init__(self, passes: List[Pass], lr_h: int, lr_w: int, scale: int,
                 win_w: int, win_h: int, title: str = "SR Video (GL)",
                 fullscreen: bool = False, visible: bool = True):
        self.passes = passes
        self.lr_h, self.lr_w, self.scale = lr_h, lr_w, scale
        self.hr_h, self.hr_w = lr_h * scale, lr_w * scale

        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.VISIBLE, GL.GL_TRUE if visible else GL.GL_FALSE)
        if not visible:
            fullscreen = False
        self.window = glfw.create_window(win_w, win_h, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # no vsync: don't cap the FPS we're measuring

        self._fullscreen = False
        self._windowed_rect = (0, 0, win_w, win_h)

        max_units = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_IMAGE_UNITS))
        need = max((len(p.binds) for p in passes), default=0)
        if need > max_units:
            raise RuntimeError(f"A pass needs {need} texture units but only {max_units} available; "
                               f"lower chunk_size in build_passes().")

        self._vao = GL.glGenVertexArrays(1)  # core profile requires a bound VAO for draws

        # Input texture (LR frame) + CUDA registration; alpha kept 0 so the padded conv column is inert.
        self._in_tex = self._new_texture(lr_h, lr_w, "rgba8")
        err, self._in_res = rt.cudaGraphicsGLRegisterImage(
            int(self._in_tex), int(GL.GL_TEXTURE_2D),
            rt.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)
        _cuda_check(err, "cudaGraphicsGLRegisterImage(input)")
        self._in_rgba = torch.zeros((lr_h, lr_w, 4), dtype=torch.uint8, device="cuda")

        self._textures: Dict[str, int] = {INPUT: self._in_tex}
        self._fbos: Dict[str, int] = {}
        self._programs: List[dict] = []
        self._final_key: Optional[str] = None
        self._build_graph()

        self._display_prog = self._link(_DISPLAY_VERT, _DISPLAY_FRAG)
        GL.glUseProgram(self._display_prog)
        GL.glUniform1i(GL.glGetUniformLocation(self._display_prog, b"tex"), 0)

        if fullscreen:
            self.toggle_fullscreen()

    # -- GL object helpers ---------------------------------------------------------------
    @staticmethod
    def _compile(src: str, stage) -> int:
        sh = GL.glCreateShader(stage)
        GL.glShaderSource(sh, src)
        GL.glCompileShader(sh)
        if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
            raise RuntimeError(f"shader compile failed:\n{GL.glGetShaderInfoLog(sh).decode()}\n---\n{src}")
        return sh

    def _link(self, vert: str, frag: str) -> int:
        vs = self._compile(vert, GL.GL_VERTEX_SHADER)
        fs = self._compile(frag, GL.GL_FRAGMENT_SHADER)
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(prog).decode())
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return prog

    def _new_texture(self, h: int, w: int, fmt: str) -> int:
        internal, gformat, gtype = ((GL.GL_RGBA16F, GL.GL_RGBA, GL.GL_FLOAT) if fmt == "f16"
                                    else (GL.GL_RGBA8, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE))
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, w, h, 0, gformat, gtype, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return tex

    def _build_graph(self):
        for p in self.passes:
            out_h, out_w = self.lr_h * p.out_scale, self.lr_w * p.out_scale
            tex = self._new_texture(out_h, out_w, p.fmt)
            self._textures[p.save] = tex
            if p.is_final:
                self._final_key = p.save

            fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                      GL.GL_TEXTURE_2D, tex, 0)
            if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"incomplete FBO for pass {p.save}")
            self._fbos[p.save] = fbo

            prog = self._link(VERT_SRC, p.frag)
            GL.glUseProgram(prog)
            # Assign each bound texture a fixed unit and wire up the sampler uniform once.
            units = []
            for unit, (uniform, src_key) in enumerate(p.binds):
                loc = GL.glGetUniformLocation(prog, uniform.encode())
                GL.glUniform1i(loc, unit)
                units.append((unit, src_key))
            # Resolution uniforms are constant for the session -> set once here.
            self._set_if_present(prog, "in_texel", (1.0 / self.lr_w, 1.0 / self.lr_h))
            self._set_if_present(prog, "lr_res", (float(self.lr_w), float(self.lr_h)))
            self._set_if_present(prog, "hr_res", (float(out_w), float(out_h)))

            self._programs.append({"prog": prog, "fbo": fbo, "units": units,
                                   "vp": (out_w, out_h)})
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    @staticmethod
    def _set_if_present(prog: int, name: str, xy: Tuple[float, float]):
        loc = GL.glGetUniformLocation(prog, name.encode())
        if loc != -1:
            GL.glUniform2f(loc, xy[0], xy[1])

    # -- per-frame -----------------------------------------------------------------------
    def upload_input(self, frame_chw_uint8: torch.Tensor):
        """Copy a (3,H,W) uint8 RGB CUDA tensor into the input GL texture, device->device."""
        self._in_rgba[..., :3] = frame_chw_uint8.permute(1, 2, 0)
        _cuda_check(rt.cudaGraphicsMapResources(1, self._in_res, 0), "map(input)")
        err, array = rt.cudaGraphicsSubResourceGetMappedArray(self._in_res, 0, 0)
        _cuda_check(err, "getMappedArray(input)")
        row_bytes = self.lr_w * 4
        _cuda_check(rt.cudaMemcpy2DToArray(
            array, 0, 0, int(self._in_rgba.data_ptr()), row_bytes, row_bytes, self.lr_h,
            rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice), "memcpy2DToArray(input)")
        _cuda_check(rt.cudaGraphicsUnmapResources(1, self._in_res, 0), "unmap(input)")

    def run_passes(self):
        """Execute every render pass; the final HR frame ends up in the FINAL texture."""
        GL.glBindVertexArray(self._vao)
        for p in self._programs:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, p["fbo"])
            GL.glViewport(0, 0, p["vp"][0], p["vp"][1])
            GL.glUseProgram(p["prog"])
            for unit, src_key in p["units"]:
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._textures[src_key])
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindVertexArray(0)

    def infer(self, frame_chw_uint8: torch.Tensor):
        self.upload_input(frame_chw_uint8)
        self.run_passes()

    def draw_final(self):
        """Draw the FINAL HR texture into the (default) framebuffer, letterboxed. No buffer swap."""
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if fb_w > 0 and fb_h > 0:
            scale = min(fb_w / self.hr_w, fb_h / self.hr_h)
            vw, vh = int(self.hr_w * scale), int(self.hr_h * scale)
            GL.glViewport((fb_w - vw) // 2, (fb_h - vh) // 2, vw, vh)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self._display_prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._textures[self._final_key])
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)

    def present(self, title: Optional[str] = None):
        """Draw the FINAL HR texture to the window, letterboxed, and swap buffers."""
        self.draw_final()
        if title is not None:
            glfw.set_window_title(self.window, title)
        glfw.swap_buffers(self.window)

    def finish(self):
        """Block until all issued GL commands have completed (for timing shader work)."""
        GL.glFinish()

    def read_final(self) -> np.ndarray:
        """Read the FINAL HR texture back to a (H,W,3) uint8 numpy array (row 0 = image top)."""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbos[self._final_key])
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        buf = GL.glReadPixels(0, 0, self.hr_w, self.hr_h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.hr_h, self.hr_w, 3)

    # -- window / input ------------------------------------------------------------------
    def poll(self):
        glfw.poll_events()

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)

    def set_key_callback(self, cb):
        glfw.set_key_callback(self.window, cb)

    def toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            x, y = glfw.get_window_pos(self.window)
            w, h = glfw.get_window_size(self.window)
            self._windowed_rect = (x, y, w, h)
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(self.window, monitor, 0, 0,
                                    mode.size.width, mode.size.height, mode.refresh_rate)
        else:
            x, y, w, h = self._windowed_rect
            glfw.set_window_monitor(self.window, None, x, y, w, h, 0)
        glfw.swap_interval(0)

    def close(self):
        if self._in_res is not None:
            rt.cudaGraphicsUnregisterResource(self._in_res)
            self._in_res = None
        try:
            glfw.destroy_window(self.window)
            glfw.terminate()
        except Exception:
            pass
