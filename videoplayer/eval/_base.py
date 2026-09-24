from pathlib import Path


class _BaseVideoPerfEvaluator:
    """
    Shared plumbing for the per-frame video SR speed evaluators: the common constructor, the fixed
    reference screen used to derive a monitor-independent display-downscale target, and the
    decode/sr/display/total summary. Subclasses implement `evaluate()` (the decode + SR + display
    timing loop differs per pipeline) and return `self._summarize(totals)`.
    """

    # Reference screen used to derive the display downscale target, so the "display" measurement is
    # deterministic and independent of whatever monitor the eval happens to run on.
    _ref_screen = (1080, 1920)

    def __init__(self, checkpoint_path, name, video_path, upscale_factor=2,
                 warmup_runs=20, iterations=200):
        self.checkpoint_path = Path(checkpoint_path)
        self.name = name
        self.video_path = Path(video_path)
        self.upscale_factor = upscale_factor
        self.warmup_runs = warmup_runs
        self.iterations = iterations

    def _summarize(self, totals):
        decode = totals["decode"] / self.iterations * 1000
        sr = totals["sr"] / self.iterations * 1000
        display = totals["display"] / self.iterations * 1000
        total = decode + sr + display

        print("-" * 30)
        print(f"decode  : {decode:8.2f} ms")
        print(f"sr      : {sr:8.2f} ms")
        print(f"display : {display:8.2f} ms")
        print(f"total   : {total:8.2f} ms")
        print("-" * 30)

        return {"decode": f"{decode:.2f}", "sr": f"{sr:.2f}",
                "display": f"{display:.2f}", "total": f"{total:.2f}"}
