"""Conservative GPU installation profiles; no scientific imports in QGIS."""

import csv
import io
import json
import os
import platform
import re
import shutil

from .client import EngineError, run_process

# Keep framework pairs explicit. TensorFlow's and-cuda extra pins a different
# CUDA stack, so managed GPU profiles accelerate SAM and keep StarDist on CPU.
# Pair/builds: https://pytorch.org/get-started/previous-versions/
# Driver floors: https://docs.nvidia.com/cuda/archive/12.9.2/cuda-toolkit-release-notes/index.html
TORCH = "2.10.0"
TORCHVISION = "0.25.0"
TENSORFLOW = "2.18.1"
NUMPY = "1.26.4"


def platform_error(device):
    system, machine = platform.system(), platform.machine().lower()
    if device == "cpu":
        return ""
    if device == "cuda":
        if (system, machine) != ("Linux", "x86_64"):
            return "Automatic NVIDIA setup requires Linux x64. Use CPU or a prepared environment on this platform."
    elif device == "mps":
        if (system, machine) != ("Darwin", "arm64"):
            return "Automatic Apple GPU setup requires native Apple Silicon macOS. Intel and Rosetta installations are not supported."
        version = platform.mac_ver()[0].split(".")[0]
        if not version.isdigit() or int(version) < 14:
            return "Automatic Apple GPU setup requires macOS 14 or later."
    else:
        return "Choose CPU, NVIDIA CUDA, or Apple MPS."
    return ""


def installation_profile(device, *, cancel=None):
    """Check prerequisites before downloading; never install system drivers."""
    error = platform_error(device)
    if error:
        raise EngineError(error)
    profile = {"device": device, "refinement": "cpu", "index": "", "hardware": ""}
    if device != "cuda":
        return profile
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1" or (
        "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"].strip()
    ):
        raise EngineError("CUDA_VISIBLE_DEVICES hides the GPU. Change it and restart QGIS, or use CPU.")
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise EngineError(
            "NVIDIA driver tools were not found (nvidia-smi). Install a supported "
            "driver using your operating system's instructions, restart, and retry; or use CPU."
        )
    code, output = run_process(
        [smi, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader,nounits"],
        cancel=cancel, timeout=20,
    )
    cards = []
    for row in csv.reader(io.StringIO(output)):
        if len(row) != 3:
            continue
        name, driver, capability = (value.strip() for value in row)
        if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", driver) or not re.fullmatch(r"\d+\.\d+", capability):
            continue
        driver_version = tuple(int(part) for part in driver.split("."))
        cards.append((name, driver, driver_version + (0,) * (3 - len(driver_version)),
                      tuple(int(part) for part in capability.split("."))))
    if code or not cards:
        raise EngineError(
            "Could not identify an NVIDIA GPU and driver. Check nvidia-smi or use CPU.\n" + output[-2000:]
        )
    # Use the newer build for Blackwell; the final computation test is still
    # required, including for mixed cards and CUDA_VISIBLE_DEVICES selections.
    newer = any(card[3] >= (10, 0) for card in cards)
    # Full toolkit driver floors (12.8 update 1 / 12.6 update 3), intentionally
    # stricter than CUDA minor-version compatibility with older drivers.
    minimum = (570, 124, 6) if newer else (560, 35, 5)
    if any(card[2] < minimum for card in cards):
        required = ".".join(str(part) for part in minimum)
        raise EngineError(
            f"This GPU profile requires NVIDIA driver {required} or newer. "
            "Update the driver through your operating system, or use CPU. "
            "The plugin does not change drivers."
        )
    if any(card[3] < (5, 0) for card in cards):
        raise EngineError("This GPU profile does not support the detected older NVIDIA architecture. Use CPU or a prepared environment.")
    profile.update(
        index="https://download.pytorch.org/whl/" + ("cu128" if newer else "cu126"),
        hardware="; ".join(f"{name} (driver {driver})" for name, driver, _, _ in cards),
    )
    return profile


def check_interpreter_platform(python, *, cancel=None):
    """An arm64 QGIS can otherwise accidentally create an Intel Python venv."""
    code, output = run_process(
        [python, "-I", "-c", "import json,platform; print(json.dumps([platform.system(),platform.machine().lower()]))"],
        python=python, cancel=cancel, timeout=30,
    )
    try:
        actual = json.loads(output)
    except ValueError:
        actual = None
    expected = [platform.system(), platform.machine().lower()]
    if code or actual != expected:
        raise EngineError(
            f"GPU setup needs a native {' '.join(expected)} Python 3.10–3.12. "
            "Choose a matching interpreter under Advanced."
        )
