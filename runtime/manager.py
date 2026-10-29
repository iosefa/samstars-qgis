"""Install a private venv or validate a user-selected SAMStars environment."""

import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

from .client import EngineError, call_with_python, run_process
from .worker import SAMSTARS_VERSION

SETTING = "samstars/engine_python"


def engine_python():
    from qgis.core import QgsSettings

    value = os.environ.get("SAMSTARS_PYTHON") or QgsSettings().value(
        SETTING, "", type=str
    )
    return str(Path(value).expanduser().absolute()) if value else None


def suggested_python():
    for name in ("python3.12", "python3.11", "python3.10"):
        found = shutil.which(name)
        if found:
            return found
    candidate = Path(sys.prefix) / ("python.exe" if os.name == "nt" else "bin/python3")
    return str(candidate) if candidate.is_file() else ""


def check_python(python, *, cancel=None):
    python = str(Path(python).expanduser().absolute())
    if not Path(python).is_file():
        raise EngineError("Choose the Python executable inside a Python 3.10–3.12 environment.")
    code, output = run_process(
        [python, "-I", "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
        python=python, cancel=cancel, timeout=30,
    )
    try:
        version = tuple(json.loads(output))
    except (ValueError, TypeError):
        version = ()
    if code or version not in ((3, 10), (3, 11), (3, 12)):
        raise EngineError("SAMStars needs Python 3.10, 3.11, or 3.12. Choose a supported interpreter.")
    return python


def check_engine(python, *, event=None, cancel=None):
    python = check_python(python, cancel=cancel)
    result = call_with_python(python, "check", {}, cancel=cancel, event=event, timeout=300)
    if not result.get("ready"):
        failures = [f"{item['name']}: {item['error']}" for item in result["checks"] if not item["ok"]]
        raise EngineError("The SAMStars environment is not ready.\n" + "\n".join(failures))
    return {**result, "python": python}


def install_engine(base_python, root, *, event=None, cancel=None):
    """Only return a new interpreter after validating it; preserve existing engines."""
    supported = (platform.system(), platform.machine().lower()) in {
        ("Linux", "x86_64"), ("Darwin", "arm64"),
    }
    if not supported:
        raise EngineError(
            "Automatic installation supports Linux x64 and Apple Silicon macOS. "
            "Select an existing SAMStars environment on other platforms."
        )
    base_python = check_python(base_python, cancel=cancel)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / ("engine-" + uuid.uuid4().hex)
    destination.mkdir()
    python = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def log(message):
        if event:
            event({"type": "log", "message": message})

    def execute(command, interpreter):
        code, output = run_process(
            command, python=interpreter, cancel=cancel, event=log,
        )
        if code:
            raise EngineError(f"Engine setup failed (exit {code}).\n{output[-4000:]}")

    try:
        log("Creating an isolated Python environment…")
        execute([base_python, "-I", "-m", "venv", str(destination)], base_python)
        pip = [str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
        if platform.system() == "Linux":
            log("Installing CPU builds of PyTorch and Torchvision…")
            execute(pip + ["torch>=2.3,<3", "torchvision>=0.18,<1", "--index-url", "https://download.pytorch.org/whl/cpu"], python)
        log(f"Installing SAMStars {SAMSTARS_VERSION} and its dependencies…")
        execute(pip + [f"samstars=={SAMSTARS_VERSION}"], python)
        log("Checking imports and CPU operations…")
        return check_engine(str(python), event=event, cancel=cancel)
    except BaseException:
        # This unique directory was created by this invocation only.
        shutil.rmtree(destination, ignore_errors=True)
        raise
