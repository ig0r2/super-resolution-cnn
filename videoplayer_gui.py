"""
Launcher GUI for the SR video player: pick the architecture and model, runtime / decoder / display and
the video, then start videoplayer/run.py in its own process (so the player's cv2 / OpenGL window and its
CUDA context never share a process with Tk, and a crash in the player leaves the launcher open). The
player's output is streamed into the log pane.

    python videoplayer_gui.py

Deliberately does not import torch / the players: it only builds the run.py command line, so it opens
instantly. Runtimes and decoders are offered only if their packages are installed (and, for onnxruntime,
if the execution provider is available); run.py (players/factory.check_combination) stays the authority
on what combinations work.
"""
import importlib.util
import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path, PurePath
from tkinter import filedialog, ttk

ROOT = Path(__file__).resolve().parent
RUN_SCRIPT = ROOT / "videoplayer" / "run.py"
MODEL_DIRS = (ROOT / "exports" / "pth", ROOT / "checkpoints")
VIDEO_DIR = ROOT / "videoinput"
SETTINGS = Path.home() / ".sr_videoplayer_gui.json"
ALL_ARCHS = "(all)"

# Decoders / displays each runtime can take, mirroring players/factory._COMBINATIONS.
_CPU_RUNTIME = {"decoder": ("pyav", "cv2"), "display": ("cv2",)}
COMBINATIONS = {
    "tensorrt": {"decoder": ("nvdec", "pyav", "cv2"), "display": ("cv2", "opengl")},
    "tensorrt-pt2": _CPU_RUNTIME,
    "ncnn-vulkan": _CPU_RUNTIME,
    "onnxruntime-cuda": _CPU_RUNTIME,
    "onnxruntime-tensorrt": _CPU_RUNTIME,
    "onnxruntime-openvino": _CPU_RUNTIME,
    "onnxruntime-directml": _CPU_RUNTIME,
    "onnxruntime-cpu": _CPU_RUNTIME,
    "opengl": {"decoder": ("nvdec", "pyav", "cv2"), "display": ("opengl",)},
}

# Packages each runtime / decoder / GPU display path imports.
RUNTIME_PACKAGES = {"tensorrt": ("tensorrt",), "tensorrt-pt2": ("torch_tensorrt",),
                    "ncnn-vulkan": ("ncnn", "pnnx"), "opengl": ("glfw", "OpenGL")}
ORT_PROVIDERS = {"onnxruntime-cuda": "CUDAExecutionProvider", "onnxruntime-tensorrt": "TensorrtExecutionProvider",
                 "onnxruntime-openvino": "OpenVINOExecutionProvider", "onnxruntime-directml": "DmlExecutionProvider",
                 "onnxruntime-cpu": "CPUExecutionProvider"}
DECODER_PACKAGES = {"nvdec": ("PyNvVideoCodec",), "pyav": ("av",), "cv2": ("cv2",)}
GL_DISPLAY_PACKAGES = ("glfw", "OpenGL", "cuda.bindings")  # tensorrt + nvdec + opengl display (CUDA-GL interop)

DEFAULTS = {
    "arch": "FastEDSR",
    "model": "multiscale/SR_FastEDSR_4_128",
    "video": str(VIDEO_DIR / "ldv-001-480p.mp4"),
    "runtime": "tensorrt",
    "decoder": "nvdec",
    "display": "cv2",
    "scales": "2 3 4",
    "audio": True,
    "fullscreen": True,
}


def _missing(packages):
    """Names of the packages that can't be imported."""
    missing = []
    for p in packages:
        try:
            found = importlib.util.find_spec(p) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(p)
    return missing


def detect_unavailable():
    """{runtime / decoder: reason} for everything this machine can't run."""
    out = {}
    for rt, pkgs in RUNTIME_PACKAGES.items():
        if missing := _missing(pkgs):
            out[rt] = "missing " + ", ".join(missing)
    try:
        import onnxruntime
        providers = set(onnxruntime.get_available_providers())
        for rt, provider in ORT_PROVIDERS.items():
            if provider not in providers:
                out[rt] = f"no {provider}"
    except Exception as e:  # not installed or a broken install
        out.update({rt: f"onnxruntime: {e.__class__.__name__}" for rt in ORT_PROVIDERS})
    for dec, pkgs in DECODER_PACKAGES.items():
        if missing := _missing(pkgs):
            out[dec] = "missing " + ", ".join(missing)
    return out


def find_models():
    """Checkpoint names ("multiscale/SR_FastEDSR_4_128") found under exports/pth/ or checkpoints/."""
    names = set()
    for d in MODEL_DIRS:
        if d.is_dir():
            names.update(p.relative_to(d).with_suffix("").as_posix() for p in d.rglob("*.pth"))
    return sorted(names, key=str.lower)


def arch_of(model):
    """"2x/SR_ABPN_2x_4_128" -> "ABPN" (the token after SR_ in the checkpoint name)."""
    parts = PurePath(model).stem.split("_")
    return parts[1] if len(parts) > 1 and parts[0] == "SR" else "other"


def gl_capable(model):
    """The opengl runtime only runs SR_FastEDSR_Multi, which are the multiscale/SR_FastEDSR_* checkpoints."""
    p = PurePath(model)
    return p.parent.name == "multiscale" and arch_of(model) == "FastEDSR"


def load_settings():
    try:
        return {**DEFAULTS, **json.loads(SETTINGS.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


class LauncherGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc = None
        self.lines = queue.Queue()
        self.models = find_models()
        self.archs = sorted({arch_of(m) for m in self.models}, key=str.lower)
        self.unavailable = detect_unavailable()
        s = load_settings()

        self.arch = tk.StringVar(value=s["arch"] if s["arch"] in self.archs else ALL_ARCHS)
        self.model = tk.StringVar(value=s["model"])
        self.model_filter = tk.StringVar()
        self.video = tk.StringVar(value=s["video"])
        self.runtime = tk.StringVar(value=s["runtime"])
        self.decoder = tk.StringVar(value=s["decoder"])
        self.display = tk.StringVar(value=s["display"])
        self.scales = tk.StringVar(value=s["scales"])
        self.audio = tk.BooleanVar(value=s["audio"])
        self.fullscreen = tk.BooleanVar(value=s["fullscreen"])

        root.title("SR Video Player")
        root.minsize(640, 480)
        self._build()
        self._apply_filter()
        self._update_runtimes()
        self.model.trace_add("write", lambda *_: self._update_runtimes())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._drain_log)

    # ---------------------------------------------------------------- layout
    def _build(self):
        f = ttk.Frame(self.root, padding=10)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(f, text="Video").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(f, textvariable=self.video).grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Button(f, text="Browse...", command=self._browse_video).grid(row=row, column=2)
        row += 1

        ttk.Label(f, text="Architecture").grid(row=row, column=0, sticky="w", pady=3)
        arch = ttk.Combobox(f, textvariable=self.arch, values=[ALL_ARCHS, *self.archs], state="readonly")
        arch.grid(row=row, column=1, sticky="w", padx=5)
        arch.bind("<<ComboboxSelected>>", lambda _: self._apply_filter())
        row += 1

        ttk.Label(f, text="Filter").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(f, textvariable=self.model_filter).grid(row=row, column=1, sticky="ew", padx=5)
        self.model_filter.trace_add("write", lambda *_: self._apply_filter())
        self.count = ttk.Label(f)
        self.count.grid(row=row, column=2)
        row += 1

        ttk.Label(f, text="Model").grid(row=row, column=0, sticky="w", pady=3)
        self.model_box = ttk.Combobox(f, textvariable=self.model)
        self.model_box.grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Button(f, text="Browse...", command=self._browse_model).grid(row=row, column=2)
        row += 1

        self.runtime_menu = self._choice(f, row, "Runtime", self.runtime)
        row += 1
        self.decoder_menu = self._choice(f, row, "Decoder", self.decoder)
        row += 1
        self.display_menu = self._choice(f, row, "Display", self.display)
        row += 1

        ttk.Label(f, text="Scales").grid(row=row, column=0, sticky="w", pady=3)
        opts = ttk.Frame(f)
        opts.grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
        ttk.Entry(opts, textvariable=self.scales, width=10).pack(side="left")
        ttk.Checkbutton(opts, text="Audio", variable=self.audio).pack(side="left", padx=(15, 0))
        ttk.Checkbutton(opts, text="Fullscreen", variable=self.fullscreen).pack(side="left", padx=(15, 0))
        row += 1

        self.hint = ttk.Label(f, foreground="gray", wraplength=600, justify="left")
        self.hint.grid(row=row, column=0, columnspan=3, sticky="w", pady=(3, 0))
        row += 1

        buttons = ttk.Frame(f)
        buttons.grid(row=row, column=0, columnspan=3, sticky="w", pady=8)
        self.play_btn = ttk.Button(buttons, text="Play", command=self._play)
        self.play_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        ttk.Button(buttons, text="Clear log", command=lambda: self.log.delete("1.0", "end")).pack(side="left")
        row += 1

        f.rowconfigure(row, weight=1)
        log_frame = ttk.Frame(f)
        log_frame.grid(row=row, column=0, columnspan=3, sticky="nsew")
        self.log = tk.Text(log_frame, height=12, wrap="none", font=("Consolas", 9))
        sb = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

    @staticmethod
    def _choice(parent, row, label, var):
        """Drop-down whose entries can be greyed out (a ttk.Combobox can't disable single items)."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        button = ttk.Menubutton(parent, textvariable=var, width=24)
        button.grid(row=row, column=1, sticky="w", padx=5)
        menu = tk.Menu(button, tearoff=False)
        button["menu"] = menu
        return menu

    def _fill(self, menu, var, items):
        """items: [(value, reason it can't be used or None)]. Unusable ones are listed greyed out with
        the reason; if the current value is unusable, the first usable one is selected."""
        menu.delete(0, "end")
        for value, reason in items:
            menu.add_radiobutton(label=value if reason is None else f"{value}   ({reason})", value=value,
                                 variable=var, command=self._on_runtime,
                                 state="normal" if reason is None else "disabled")
        usable = [v for v, reason in items if reason is None]
        if var.get() not in usable and usable:
            var.set(usable[0])

    # ---------------------------------------------------------------- choices
    def _apply_filter(self):
        """Models of the selected architecture that contain every word of the filter."""
        arch = self.arch.get()
        words = self.model_filter.get().lower().split()
        models = [m for m in self.models
                  if arch in (ALL_ARCHS, arch_of(m)) and all(w in m.lower() for w in words)]
        self.model_box["values"] = models
        self.count["text"] = f"{len(models)} models"
        if models and self.model.get() not in models:
            self.model.set(models[0])

    def _update_runtimes(self):
        """All runtimes are listed; the ones this machine can't run (checked at startup), and opengl
        for models it can't run, are greyed out with the reason."""
        model_ok = gl_capable(self.model.get())
        self._fill(self.runtime_menu, self.runtime, [
            (rt, self.unavailable.get(rt) or (None if rt != "opengl" or model_ok else "only multiscale/SR_FastEDSR_*"))
            for rt in COMBINATIONS])
        self._on_runtime()

    def _on_runtime(self):
        """List the decoders / displays, greying out what the selected runtime (and decoder) can't take."""
        rt = self.runtime.get()
        combo = COMBINATIONS.get(rt, _CPU_RUNTIME)
        self._fill(self.decoder_menu, self.decoder, [
            (d, self.unavailable.get(d) or (None if d in combo["decoder"] else f"not with {rt}"))
            for d in DECODER_PACKAGES])

        gl_missing = _missing(GL_DISPLAY_PACKAGES)

        def display_reason(d):
            if d not in combo["display"]:
                return f"not with {rt}"
            if d == "opengl" and rt == "tensorrt":  # needs NVDEC frames and CUDA-GL interop
                if self.decoder.get() != "nvdec":
                    return "needs decoder nvdec"
                if gl_missing:
                    return "missing " + ", ".join(gl_missing)
            return None

        self._fill(self.display_menu, self.display, [(d, display_reason(d)) for d in ("cv2", "opengl")])

        self.hint["text"] = ("First run of a model / resolution builds the TensorRT engine (can take a while)."
                             if rt == "tensorrt" else "")

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Video", initialdir=VIDEO_DIR if VIDEO_DIR.is_dir() else ROOT,
            filetypes=[("Video", "*.mp4 *.mkv *.webm *.avi *.mov *.y4m"), ("All files", "*.*")])
        if path:
            self.video.set(path)

    def _browse_model(self):
        path = filedialog.askopenfilename(title="Checkpoint", initialdir=MODEL_DIRS[0],
                                          filetypes=[("Checkpoint", "*.pth")])
        if path:
            self.model.set(path)

    # ---------------------------------------------------------------- run
    def _settings(self):
        return {"arch": self.arch.get(), "model": self.model.get().strip(), "video": self.video.get().strip(),
                "runtime": self.runtime.get(), "decoder": self.decoder.get(), "display": self.display.get(),
                "scales": self.scales.get(), "audio": self.audio.get(), "fullscreen": self.fullscreen.get()}

    def _play(self):
        if self.proc is not None:
            return
        s = self._settings()
        if not Path(s["video"]).is_file():
            self._write(f"Video not found: {s['video']}\n")
            return
        try:
            scales = [str(int(x)) for x in s["scales"].replace(",", " ").split()]
        except ValueError:
            self._write(f"Bad scales {s['scales']!r} (expected e.g. '2 3 4')\n")
            return
        try:
            SETTINGS.write_text(json.dumps(s, indent=2), encoding="utf-8")
        except OSError:
            pass

        cmd = [sys.executable, "-u", str(RUN_SCRIPT), "--video", s["video"], "--model", s["model"],
               "--runtime", s["runtime"], "--decoder", s["decoder"], "--display", s["display"],
               "--scales", *scales]
        if not s["audio"]:
            cmd.append("--no-audio")
        if not s["fullscreen"]:
            cmd.append("--windowed")

        self._write("> " + subprocess.list2cmdline(cmd[2:]) + "\n")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        self.play_btn["state"] = "disabled"
        self.stop_btn["state"] = "normal"

    def _pump(self, proc):
        for line in proc.stdout:
            self.lines.put(line)
        self.lines.put(proc.wait())  # int = exit code, marks the end

    def _drain_log(self):
        try:
            while True:
                item = self.lines.get_nowait()
                if isinstance(item, int):
                    self._write(f"[player exited with code {item}]\n")
                    self.proc = None
                    self.play_btn["state"] = "normal"
                    self.stop_btn["state"] = "disabled"
                else:
                    self._write(item)
        except queue.Empty:
            pass
        self.root.after(50, self._drain_log)

    def _write(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _stop(self):
        if self.proc is not None:
            self.proc.terminate()

    def _on_close(self):
        self._stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    LauncherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
