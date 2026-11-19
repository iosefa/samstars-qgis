"""Run SAMStars outside QGIS.

Adapted from PyForestScan for QGIS (GPL-2.0-or-later).
"""

from collections import deque
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time

from .progress import TrainingProgress, clean_log

PROTOCOL = 1


class EngineError(RuntimeError):
    """An actionable engine setup or processing failure."""


class EngineCanceled(EngineError):
    pass


def sanitized_environment(python=None):
    """Keep user/network essentials; rebuild executable and geospatial paths.

    In particular, never inherit QGIS's Python, Qt, GDAL/PROJ, conda or dynamic
    library overrides. No shell activation or configuration changes are used.
    """
    allowed = {
        "HOME",
        "USER",
        "LOGNAME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LANGUAGE",
        "TZ",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "PIP_CERT",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed or key.upper().startswith("LC_")
    }
    if os.name == "nt":
        system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        paths = [system / "System32", system]
    else:
        paths = [Path("/usr/bin"), Path("/bin"), Path("/usr/sbin"), Path("/sbin")]
    if python is not None:
        python = Path(python).absolute()
        prefix = (
            python.parent.parent
            if os.name != "nt" or python.parent.name.lower() == "scripts"
            else python.parent
        )
        if os.name == "nt":
            paths = [prefix, prefix / "Library" / "bin", prefix / "Scripts"] + paths
            share = prefix / "Library" / "share"
        else:
            paths = [prefix / "bin"] + paths
            share = prefix / "share"
        if (share / "gdal").is_dir():
            env["GDAL_DATA"] = str(share / "gdal")
        if (share / "proj").is_dir():
            env["PROJ_DATA"] = str(share / "proj")
            env["PROJ_LIB"] = str(share / "proj")
    env["PATH"] = os.pathsep.join(str(path) for path in paths)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _stop_process(process):
    if os.name == "nt" and process.poll() is not None:
        return
    if os.name == "nt":
        taskkill = (
            Path(os.environ.get("SystemRoot", r"C:\Windows"))
            / "System32"
            / "taskkill.exe"
        )
        try:
            subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
    finally:
        if os.name != "nt":
            # The worker may exit before its TensorFlow subprocess. Stop the
            # entire group even if the group leader has already finished.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait(timeout=5)


def run_process(command, *, python=None, cancel=None, event=None, timeout=None):
    """Drain both pipes on reader threads so cancellation works on Windows too."""
    if cancel and cancel():
        raise EngineCanceled("SAMStars processing was canceled.")
    options = (
        {"start_new_session": True}
        if os.name != "nt"
        else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        }
    )
    try:
        process = subprocess.Popen(
            [str(value) for value in command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=sanitized_environment(python),
            **options,
        )
    except OSError as exc:
        raise EngineError(
            f"Could not start the SAMStars engine: {exc}. Open SAMStars Setup to repair it."
        ) from exc
    messages = queue.Queue()
    tail = deque(maxlen=40)

    def reader(stream):
        try:
            for line in stream:
                messages.put(line.rstrip("\r\n"))
        finally:
            stream.close()
            messages.put(None)

    threads = [
        threading.Thread(target=reader, args=(stream,), daemon=True)
        for stream in (process.stdout, process.stderr)
    ]
    for thread in threads:
        thread.start()
    started = time.monotonic()
    finished = 0
    try:
        while finished < 2 or process.poll() is None:
            if cancel and cancel():
                raise EngineCanceled("SAMStars processing was canceled.")
            if timeout is not None and time.monotonic() - started > timeout:
                raise EngineError(
                    "The SAMStars engine timed out. Open SAMStars Setup to repair or retry."
                )
            try:
                line = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                finished += 1
                continue
            if line:
                tail.append(line[-4000:])
                if event:
                    event(line)
        return process.wait(), "\n".join(tail)
    finally:
        _stop_process(process)
        for thread in threads:
            thread.join(timeout=1)


def call_with_python(
    python, operation, payload, feedback=None, *, cancel=None, event=None, timeout=None
):
    """Internal entry point also used to validate an unactivated installation."""
    worker = Path(__file__).resolve().with_name("worker.py")
    if not worker.is_file():
        raise EngineError(
            "The plugin is missing its engine worker. Reinstall the SAMStars plugin ZIP."
        )

    def canceled():
        return bool(
            (cancel and cancel()) or (feedback is not None and feedback.isCanceled())
        )

    training_progress = TrainingProgress() if operation == "train" else None

    def consume(line):
        try:
            message = json.loads(line)
        except (ValueError, TypeError):
            message = {"type": "log", "message": line}
        if not isinstance(message, dict):
            message = {"type": "log", "message": line}
        if "message" in message:
            message["message"] = clean_log(str(message["message"]))
        if event:
            event(message)
        if feedback is not None:
            if training_progress is not None:
                status = training_progress.consume(message.get("message", ""))
                if status:
                    feedback.setProgressText(status)
            if message.get("type") == "progress":
                try:
                    feedback.setProgress(max(0, min(100, float(message["value"]))))
                except (KeyError, TypeError, ValueError):
                    pass
            else:
                feedback.pushInfo(str(message.get("message", line)))

    if canceled():
        raise EngineCanceled("SAMStars processing was canceled.")
    with tempfile.TemporaryDirectory(prefix="samstars-job-") as directory:
        request, response = (
            Path(directory) / "request.json",
            Path(directory) / "result.json",
        )
        request.write_text(
            json.dumps(
                {"protocol": PROTOCOL, "operation": operation, "payload": payload},
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        code, logs = run_process(
            [
                python,
                "-I",
                "-u",
                str(worker),
                "--request",
                str(request),
                "--result",
                str(response),
            ],
            python=python,
            cancel=canceled,
            event=consume,
            timeout=timeout,
        )
        if not response.is_file() or response.stat().st_size > 32 * 1024 * 1024:
            detail = logs[-3000:] or f"Worker exited with status {code}."
            raise EngineError(
                f"The SAMStars engine did not return a valid result. {detail}"
            )
        try:
            result = json.loads(response.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EngineError(
                "The SAMStars engine returned an unreadable result. Repair the engine in SAMStars Setup."
            ) from exc
        if not isinstance(result, dict) or result.get("protocol") != PROTOCOL:
            raise EngineError(
                "The plugin and engine use incompatible protocols. Repair the engine in SAMStars Setup."
            )
        if not result.get("ok"):
            error = result.get("error", {})
            detail = (
                error.get("message", "Unknown processing error")
                if isinstance(error, dict)
                else str(error)
            )
            raise EngineError(f"SAMStars: {detail}")
        if code != 0 or not isinstance(result.get("result"), dict):
            raise EngineError(
                f"The SAMStars engine failed with status {code}. {logs[-2000:]}"
            )
        return result["result"]


def call_engine(operation, payload, feedback=None):
    from .manager import engine_python

    python = engine_python()
    if python is None:
        raise EngineError(
            "The SAMStars engine is not installed. Open Plugins → SAMStars → Set up SAMStars to install or select it."
        )
    return call_with_python(python, operation, payload, feedback)
