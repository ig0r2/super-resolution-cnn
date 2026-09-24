"""
Zero-copy GPU display for the NVDEC pipeline via CUDA<->OpenGL interop.

The SR output already lives on the GPU as a (3,H,W) uint8 RGB tensor. Instead of copying it to
host memory and handing it to cv2.imshow, we register an OpenGL texture with CUDA once and, every
frame, copy the tensor device->device straight into that texture with cudaMemcpy2DToArray, then
draw it on a full-screen quad. No PCIe round-trip, and the display path never touches the CPU with
the pixel data.

Only one small GPU-side reshape is needed per frame: interleave the planar (3,H,W) RGB into a
packed (H,W,4) RGBA buffer (RGBA, not RGB, to keep texel rows 4-byte aligned for both the GL
texture and the CUDA copy). Everything stays on the default CUDA stream that torch uses, so the
copy is correctly ordered after the SR kernels with no explicit sync.
"""

from typing import Optional, Tuple

import glfw
import torch
from OpenGL import GL
from cuda.bindings import runtime as rt

_VERT_SRC = """
#version 330 core
layout(location = 0) in vec2 pos;
out vec2 uv;
void main() {
    // Map clip-space quad [-1,1] to texture coords, flipping Y so image row 0 is at the top.
    uv = vec2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

_FRAG_SRC = """
#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D tex;
void main() { color = texture(tex, uv); }
"""


def _cuda_check(err, msg: str = ""):
    # cuda-python runtime calls return a tuple whose first element is the status code.
    if isinstance(err, tuple):
        err = err[0]
    if err != rt.cudaError_t.cudaSuccess:
        _, name = rt.cudaGetErrorName(err)
        _, desc = rt.cudaGetErrorString(err)
        raise RuntimeError(f"CUDA error in {msg}: {name.decode()} - {desc.decode()}")


class GLDisplay:
    """A GLFW/OpenGL window that shows GPU tensors via CUDA-GL interop.

    Lifecycle: construct (opens the window + GL context), then call `upload(chw_rgb_uint8)` +
    `render()` each frame and `poll()`/`should_close()` for events, then `close()`.
    The texture and its CUDA registration are created lazily on the first upload, sized to the
    first frame (the SR output resolution is constant for a given video).
    """

    def __init__(self, win_w: int, win_h: int, title: str = "SR Video (GPU)",
                 fullscreen: bool = True, visible: bool = True):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        # Offscreen mode (visible=False) is used by the benchmark: a hidden window still gives a
        # real GL context and does the full upload/draw/swap work, without popping a window per model.
        glfw.window_hint(glfw.VISIBLE, GL.GL_TRUE if visible else GL.GL_FALSE)
        if not visible:
            fullscreen = False

        self.window = glfw.create_window(win_w, win_h, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # no vsync: don't cap the FPS we're trying to measure/maximise

        self._fullscreen = False
        self._windowed_rect = (0, 0, win_w, win_h)

        self._tex_id: Optional[int] = None
        self._resource = None
        self._tex_hw: Optional[Tuple[int, int]] = None
        self._rgba: Optional[torch.Tensor] = None  # reused packed (H,W,4) device buffer

        self._program = self._build_program()
        self._vao = self._build_quad()

        GL.glUseProgram(self._program)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, b"tex"), 0)

        if fullscreen:
            self.toggle_fullscreen()

    # -- GL setup -------------------------------------------------------------------------
    @staticmethod
    def _compile(src: str, stage) -> int:
        shader = GL.glCreateShader(stage)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(shader).decode())
        return shader

    def _build_program(self) -> int:
        vs = self._compile(_VERT_SRC, GL.GL_VERTEX_SHADER)
        fs = self._compile(_FRAG_SRC, GL.GL_FRAGMENT_SHADER)
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(prog).decode())
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        return prog

    def _build_quad(self) -> int:
        import numpy as np
        verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)  # triangle strip
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)
        return vao

    def _create_texture(self, h: int, w: int):
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        err, resource = rt.cudaGraphicsGLRegisterImage(
            int(tex), int(GL.GL_TEXTURE_2D),
            rt.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard)
        _cuda_check(err, "cudaGraphicsGLRegisterImage")

        self._tex_id = int(tex)
        self._resource = resource
        self._tex_hw = (h, w)
        self._rgba = torch.empty((h, w, 4), dtype=torch.uint8, device="cuda")

    # -- per-frame ------------------------------------------------------------------------
    def upload(self, chw_rgb_uint8: torch.Tensor):
        """Copy a (3,H,W) uint8 RGB CUDA tensor into the GL texture, device->device."""
        _, h, w = chw_rgb_uint8.shape
        if self._tex_hw != (h, w):
            if self._resource is not None:
                _cuda_check(rt.cudaGraphicsUnregisterResource(self._resource), "unregister")
                GL.glDeleteTextures(1, [self._tex_id])
            self._create_texture(h, w)

        # Planar (3,H,W) RGB -> packed (H,W,4) RGBA on the GPU (alpha is left as whatever the
        # buffer holds; the shader ignores it). Reused buffer, no allocation in steady state.
        self._rgba[..., :3] = chw_rgb_uint8.permute(1, 2, 0)

        _cuda_check(rt.cudaGraphicsMapResources(1, self._resource, 0), "map")
        err, array = rt.cudaGraphicsSubResourceGetMappedArray(self._resource, 0, 0)
        _cuda_check(err, "getMappedArray")
        row_bytes = w * 4
        _cuda_check(rt.cudaMemcpy2DToArray(
            array, 0, 0, int(self._rgba.data_ptr()), row_bytes, row_bytes, h,
            rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice), "memcpy2DToArray")
        _cuda_check(rt.cudaGraphicsUnmapResources(1, self._resource, 0), "unmap")

    def render(self):
        """Draw the current texture to the window, letterboxed to preserve aspect ratio."""
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if self._tex_hw is not None and fb_w > 0 and fb_h > 0:
            th, tw = self._tex_hw
            scale = min(fb_w / tw, fb_h / th)
            vw, vh = int(tw * scale), int(th * scale)
            GL.glViewport((fb_w - vw) // 2, (fb_h - vh) // 2, vw, vh)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if self._tex_id is not None:
            GL.glUseProgram(self._program)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
            GL.glBindVertexArray(self._vao)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            GL.glBindVertexArray(0)
        glfw.swap_buffers(self.window)

    # -- window / input -------------------------------------------------------------------
    def set_title(self, title: str):
        glfw.set_window_title(self.window, title)

    def poll(self):
        glfw.poll_events()

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)

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
        if self._resource is not None:
            rt.cudaGraphicsUnregisterResource(self._resource)
            self._resource = None
        try:
            glfw.destroy_window(self.window)
            glfw.terminate()
        except Exception:
            pass
