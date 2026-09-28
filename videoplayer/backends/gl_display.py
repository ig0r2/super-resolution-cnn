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

# Vertex shader - generates the full-screen quad from gl_VertexID
# glDrawArrays(GL_TRIANGLE_STRIP, 0, 4) runs it with gl_VertexID = 0, 1, 2, 3:
#   bit 0 -> x corner: 0, 1, 0, 1
#   bit 1 -> y corner: 0, 0, 1, 1
# giving corners (0,0), (1,0), (0,1), (1,1)
_VERT_SRC = """
#version 330 core
out vec2 uv;
void main() {
    vec2 corner = vec2(gl_VertexID & 1, (gl_VertexID >> 1) & 1);  // [0,1] per axis
    uv = vec2(corner.x, 1.0 - corner.y);    // flip Y so image row 0 is at the top
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);  // [0,1] -> clip space [-1,1], z = 0, w = 1
}
"""

# Fragment shader - takes color from texture tex on coordinate uv
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
        # Always create hidden: an unpresented window is painted white by the OS until the first
        # swap. We show it only after a black frame is in the buffer (see end of __init__).
        # Offscreen mode (visible=False) is used by the benchmark and never shows the window.
        glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
        if not visible:
            fullscreen = False

        self.window = glfw.create_window(win_w, win_h, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # no vsync: don't cap the FPS we're trying to measure/maximise

        self._fullscreen = False
        self._windowed_rect = (0, 0, win_w, win_h)  # non fullscreen window position

        self._tex_id: Optional[int] = None
        self._resource = None
        self._tex_hw: Optional[Tuple[int, int]] = None
        self._rgba: Optional[torch.Tensor] = None  # reused packed (H,W,4) device buffer

        self._program = self._build_program()
        self._vao = self._build_quad()

        GL.glUseProgram(self._program)
        GL.glUniform1i(GL.glGetUniformLocation(self._program, b"tex"), 0)

        # The window is shown by the first render() that has a frame to draw (see _show_window)
        self._pending_show = visible
        self._pending_fullscreen = fullscreen

    def _show_window(self):
        """Show the window with the first frame already in it: draw it while hidden, show,
        go fullscreen if requested, then draw again at the final framebuffer size."""
        self._pending_show = False
        self._draw()
        glfw.show_window(self.window)
        if self._pending_fullscreen:
            self.toggle_fullscreen()
        self._draw()

    ### setup ######################################
    @staticmethod
    def _compile(src: str, stage) -> int:
        shader = GL.glCreateShader(stage)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(shader).decode())
        return shader

    def _build_program(self) -> int:
        """Compiles vertex and fragment shaders and links them (out uv -> in uv) and attaches them to program"""
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
        """Creates an empty VAO for the quad.
        The vertex shader computes the 4 quad corners from gl_VertexID, so there is no vertex
        buffer (VBO) and no vertex attributes. The core profile still requires a VAO to be
        bound when drawing, so we create an empty one.
        """
        return GL.glGenVertexArrays(1)

    def _create_texture(self, h: int, w: int):
        # make new texture
        tex = GL.glGenTextures(1)
        # bind it to GL_TEXTURE_2D
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        # set bilinear for minification
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        # set bilinear for magnification
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        # no wrap
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        # allocate memory
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        # unbind texture from GL_TEXTURE_2D
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # register texture at cuda and get cuda's handle for that texture
        err, resource = (
            rt.cudaGraphicsGLRegisterImage(
                int(tex), int(GL.GL_TEXTURE_2D),
                rt.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
        _cuda_check(err, "cudaGraphicsGLRegisterImage")

        self._tex_id = int(tex)  # OpenGL texture name
        self._resource = resource  # cuda texture name (cudaGraphicsResource_t)
        self._tex_hw = (h, w)
        # make buffer for torch to write into
        self._rgba = torch.empty((h, w, 4), dtype=torch.uint8, device="cuda")

    #### per-frame ######################################
    def upload(self, chw_rgb: torch.Tensor):
        """Copy a (3,H,W) RGB CUDA tensor into the GL texture, device->device.
        uint8, or float already clamped to [0,255]: the cast to uint8 (truncating, like .to(uint8))
        is fused into the CHW->HWC copy below, so callers don't need a separate cast kernel."""
        _, h, w = chw_rgb.shape
        if self._tex_hw != (h, w):
            # if the resolution has changed (texture not fit for frame), create or recreate texture
            if self._resource is not None:
                _cuda_check(rt.cudaGraphicsUnregisterResource(self._resource), "unregister")
                GL.glDeleteTextures(1, [self._tex_id])
            self._create_texture(h, w)

        # Planar (3,H,W) RGB -> packed (H,W,4) RGBA on the GPU (alpha is left as whatever the
        # buffer holds; the shader ignores it). Reused buffer, no allocation in steady state.
        # One copy_ kernel does both the layout change and the dtype cast (if any).
        self._rgba[..., :3] = chw_rgb.permute(1, 2, 0)

        # cuda maps texture (takes over the texture from OpenGL)
        _cuda_check(rt.cudaGraphicsMapResources(1, self._resource, 0), "map")
        # get cuda handle for texture
        err, array = rt.cudaGraphicsSubResourceGetMappedArray(self._resource, 0, 0)
        _cuda_check(err, "getMappedArray")
        # copy from _rgba to array (_resource)
        row_bytes = w * 4
        _cuda_check(rt.cudaMemcpy2DToArray(
            array, 0, 0, int(self._rgba.data_ptr()),
            row_bytes, row_bytes, h, rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice), "memcpy2DToArray")
        # release resource (unmap, give it back to OpenGL)
        _cuda_check(rt.cudaGraphicsUnmapResources(1, self._resource, 0), "unmap")

    def render(self):
        """Draw the current texture to the window, letterboxed to preserve aspect ratio."""
        if self._pending_show and self._tex_id is not None:
            # show window only when there is a texture ready (to not open window before the video starts)
            self._show_window()
            return
        self._draw()

    def _draw(self):
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)  # get window resolution
        if self._tex_hw is not None and fb_w > 0 and fb_h > 0:
            th, tw = self._tex_hw
            scale = min(fb_w / tw, fb_h / th)
            vw, vh = int(tw * scale), int(th * scale)
            # set viewport (rectangle where image will go)
            GL.glViewport((fb_w - vw) // 2, (fb_h - vh) // 2, vw, vh)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if self._tex_id is not None:
            GL.glUseProgram(self._program)
            # read texture from slot 0
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)
            # draw vertices
            GL.glBindVertexArray(self._vao)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)  # 4 vertices -> gl_VertexID 0..3
            GL.glBindVertexArray(0)
        glfw.swap_buffers(self.window)

    #### window / input  ######################################
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
