"""Readable training status from the supported engines' console logs."""

from collections import deque
import re
import time

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")


def clean_log(line):
    return _ANSI.sub("", line).replace("\b", "").replace("\x00", "").strip()


def duration(seconds):
    seconds = max(1, round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = (seconds + 30) // 60
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h {minutes % 60}m"


class TrainingProgress:
    """Estimate only the current training stage, never total job completion."""

    def __init__(self):
        self.sam_started = None
        self.epoch_started = None
        self.epoch = self.total = 0
        self.epoch_times = deque(maxlen=5)

    def consume(self, line, now=None):
        now = time.monotonic() if now is None else now
        line = clean_log(line)
        if re.search(r".+:\s*epochs=\d+\s+\(iters/epoch=", line):
            self.sam_started = now
            return "SAM training started; estimating time after the first epoch."
        match = re.search(r"\bepoch (\d+)/(\d+) loss=", line)
        if match:
            done, total = map(int, match.groups())
            if not 0 < done <= total:
                return None
            status = f"SAM training: {done}/{total} epochs complete ({100 * done / total:.0f}%)."
            if done == total:
                return status + " Crown proposal generation and StarDist training follow."
            if self.sam_started is not None and now > self.sam_started:
                seconds = (now - self.sam_started) / done * (total - done)
                status += f" About {duration(seconds)} remaining in this stage."
            return status
        match = re.match(r"Epoch (\d+)/(\d+)$", line)
        if match:
            epoch, total = map(int, match.groups())
            if not 0 < epoch <= total:
                return None
            if epoch == self.epoch + 1 and total == self.total and self.epoch_started is not None:
                self.epoch_times.append(max(0, now - self.epoch_started))
            else:
                self.epoch_times.clear()
            self.epoch, self.total, self.epoch_started = epoch, total, now
            return self._stardist_status(0)
        match = re.match(r"(\d+)/(\d+)\s+.*(?:loss:|ms/step|s/step)", line)
        if match and self.epoch:
            step, total = map(int, match.groups())
            if not 0 <= step <= total or total == 0:
                return None
            if self.epoch == self.total and step == total:
                return "StarDist: final training batches complete; validation, threshold optimization, and bundle saving may remain."
            return self._stardist_status(step / total, f", step {step}/{total}")
        if "optimized thresholds:" in line or "trained refinement model ->" in line:
            return "StarDist training finished; finalizing the model bundle."
        if "SAMStars finished." in line:
            return "Training complete; the model bundle is saved."
        return None

    def _stardist_status(self, fraction, detail=""):
        done = self.epoch - 1 + fraction
        status = f"StarDist training: epoch {self.epoch}/{self.total}{detail} ({100 * done / self.total:.1f}% of training epochs)."
        if self.epoch_times:
            seconds = sum(self.epoch_times) / len(self.epoch_times) * (self.total - done)
            status += f" About {duration(seconds)} remaining in this stage."
        return status
