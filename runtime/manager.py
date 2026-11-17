"""Install a private venv or validate a user-selected SAMStars environment."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import uuid

from .client import EngineCanceled, EngineError, call_with_python, run_process
from . import accelerators
from .worker import PROFILE_FILE, SAMSTARS_VERSION

SETTING = "samstars/engine_python"
MARKER = ".samstars-engine.json"
OWNER = {"owner": "org.samstars.qgis.engine", "schema": 1}


def engine_root():
    from qgis.core import QgsApplication

    return Path(QgsApplication.qgisSettingsDirPath()) / "samstars" / "engines"


def supported_platform():
    return (platform.system(), platform.machine().lower()) in {
        ("Linux", "x86_64"), ("Darwin", "arm64"),
    }


def _owned_engine(path):
    """Only recognize directories created and marked by this installer."""
    path = Path(path)
    marker = path / MARKER
    if (
        not re.fullmatch(r"engine-[0-9a-f]{32}", path.name)
        or path.is_symlink() or not path.is_dir() or marker.is_symlink()
        or (hasattr(path, "is_junction") and path.is_junction())
    ):
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == OWNER
    except (OSError, ValueError):
        return False


def owned_engines(root):
    root = Path(root)
    if root.is_symlink() or (hasattr(root, "is_junction") and root.is_junction()):
        raise EngineError("Refusing to change a linked engine directory.")
    return sorted(path for path in root.iterdir() if _owned_engine(path)) if root.is_dir() else []


@contextmanager
def _operation_lock(root):
    """Serialize install/removal; the OS releases this lock after a crash."""
    path = Path(root) / ".setup.lock"
    if path.is_symlink():
        raise EngineError("Refusing to use a linked setup lock.")
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise EngineError("Another SAMStars setup operation is running.") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise EngineError("Another SAMStars setup operation is running.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def status():
    """Read setup state without importing scientific libraries or starting Python."""
    root = engine_root()
    python = engine_python()
    result = {
        "installed": bool(python and Path(python).is_file()),
        "python": python,
        "path": str(root),
        "platform": f"{platform.system()} {platform.machine()}",
        "managed": False,
        "removable": False,
        "override": bool(os.environ.get("SAMSTARS_PYTHON")),
        "error": "",
        "warning": "",
        "accelerator": "cpu",
    }
    try:
        generations = owned_engines(root)
        result["removable"] = bool(generations)
        result["managed"] = bool(python and any(
            Path(python).absolute().is_relative_to(path.absolute()) for path in generations
        ))
        if result["managed"]:
            profile = Path(python).parent.parent / PROFILE_FILE
            if profile.is_file():
                try:
                    saved = json.loads(profile.read_text(encoding="utf-8"))
                    if saved["device"] not in ("cpu", "cuda", "mps") or saved["refinement"] != "cpu":
                        raise ValueError("Unsupported engine profile")
                    result["accelerator"] = saved["device"]
                except (OSError, ValueError, TypeError, KeyError):
                    result["warning"] = "Private engine settings are invalid. Repair will build a CPU engine; GPU setup can select a GPU profile."
        if not supported_platform():
            result["error"] = (
                "Automatic installation supports Linux x64 and Apple Silicon macOS. "
                "Select an existing environment under Advanced on other platforms."
            )
    except (OSError, ValueError, AttributeError, EngineError) as exc:
        result["error"] = str(exc)
    return result


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


def check_gpu(python, device, *, event=None, cancel=None):
    """Run a real GPU computation outside QGIS without changing the engine."""
    if device not in ("cuda", "mps"):
        raise EngineError("Choose NVIDIA CUDA or Apple MPS to test GPU support.")
    python = check_python(python, cancel=cancel)
    result = call_with_python(
        python, "check_gpu", {"device": device}, cancel=cancel, event=event, timeout=180,
    )
    if not result.get("ready"):
        failures = [item["error"] for item in result.get("checks", []) if not item["ok"]]
        raise EngineError("GPU check failed. Use CPU or repair GPU setup.\n" + "\n".join(failures))
    return result


def install_engine(base_python=None, root=None, *, accelerator="cpu", event=None, cancel=None):
    """Only return a new interpreter after validating it; preserve existing engines."""
    if not supported_platform():
        raise EngineError(
            "Automatic installation supports Linux x64 and Apple Silicon macOS. "
            "Select an existing SAMStars environment on other platforms."
        )
    base_python = base_python or suggested_python()
    if not base_python:
        raise EngineError(
            "No Python installation was found. Open Advanced and choose a Python "
            "3.10–3.12 executable with venv support."
        )
    base_python = check_python(base_python, cancel=cancel)
    profile = accelerators.installation_profile(accelerator, cancel=cancel)
    if accelerator != "cpu":
        accelerators.check_interpreter_platform(base_python, cancel=cancel)
    root = Path(root) if root is not None else engine_root()
    owned_engines(root)  # Reject linked roots before creating anything.
    root.mkdir(parents=True, exist_ok=True)
    with _operation_lock(root):
        return _install_generation(base_python, root, profile=profile, event=event, cancel=cancel)


def _install_generation(base_python, root, *, profile, event=None, cancel=None):
    destination = root / ("engine-" + uuid.uuid4().hex)
    destination.mkdir()
    python = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def log(message, value=None):
        if event:
            event({"type": "log" if value is None else "progress", "message": message,
                   **({"value": value} if value is not None else {})})

    def execute(command, interpreter):
        code, output = run_process(
            command, python=interpreter, cancel=cancel, event=log,
        )
        if code:
            raise EngineError(f"Engine setup failed (exit {code}).\n{output[-4000:]}")

    try:
        (destination / MARKER).write_text(json.dumps(OWNER), encoding="utf-8")
        (destination / PROFILE_FILE).write_text(json.dumps(profile), encoding="utf-8")
        if profile["hardware"]:
            log("Detected: " + profile["hardware"])
        log("Creating an isolated Python environment…", 5)
        execute([base_python, "-I", "-m", "venv", str(destination)], base_python)
        pip = [str(python), "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check", "--no-input", "--no-cache-dir"]
        requirements = []
        if profile["device"] != "cpu":
            variant = "+" + profile["index"].rsplit("/", 1)[-1] if profile["index"] else ""
            torch_packages = [f"torch=={accelerators.TORCH}{variant}", f"torchvision=={accelerators.TORCHVISION}{variant}"]
            constraints = destination / "gpu-constraints.txt"
            constraints.write_text("\n".join([
                *torch_packages, f"tensorflow=={accelerators.TENSORFLOW}", f"numpy=={accelerators.NUMPY}",
            ]) + "\n", encoding="utf-8")
            requirements = ["--constraint", str(constraints), "--only-binary=:all:"]
            log("Installing GPU libraries; downloading and unpacking can take several minutes…", 15)
            execute(pip + torch_packages + ["--only-binary=:all:", "--index-url", profile["index"] or "https://pypi.org/simple"], python)
        elif platform.system() == "Linux":
            log("Installing CPU builds of PyTorch and Torchvision…", 15)
            execute(pip + ["torch>=2.3,<3", "torchvision>=0.18,<1", "--index-url", "https://download.pytorch.org/whl/cpu"], python)
        log("Installing SAMStars and its dependencies…", 45)
        execute(pip + [f"samstars=={SAMSTARS_VERSION}", "--index-url", "https://pypi.org/simple"] + requirements, python)
        execute([str(python), "-I", "-m", "pip", "--isolated", "check"], python)
        log("Checking scientific libraries and CPU operations…", 85)
        result = check_engine(str(python), event=event, cancel=cancel)
        if profile["device"] != "cpu":
            log("Testing GPU computation and gradients…", 95)
            result["gpu"] = check_gpu(str(python), profile["device"], event=event, cancel=cancel)
        if cancel and cancel():
            raise EngineCanceled("Engine setup canceled.")
        log("Engine ready.", 100)
        return {**result, "accelerator": profile["device"]}
    except BaseException:
        # This unique directory was created by this invocation only.
        shutil.rmtree(destination, ignore_errors=True)
        raise


def remove_engine(root=None, *, event=None, cancel=None):
    """Remove marked private generations, leaving other environments alone."""
    root = Path(root) if root is not None else engine_root()
    owned_engines(root)  # Reject linked roots, even when there is nothing to remove.
    if not root.is_dir():
        return {"removed": []}
    with _operation_lock(root):
        return _remove_generations(root, event=event, cancel=cancel)


def _remove_generations(root, *, event=None, cancel=None):
    generations = owned_engines(root)
    if cancel and cancel():
        raise EngineCanceled("Engine removal canceled.")
    removed = []
    for index, path in enumerate(generations):
        if not _owned_engine(path):
            raise EngineError("The engine directory changed during removal. Please retry setup.")
        try:
            shutil.rmtree(path)
        except OSError as exc:
            # Keep a partially removed generation recognizable for a later retry.
            if path.is_dir() and not path.is_symlink():
                (path / MARKER).write_text(json.dumps(OWNER), encoding="utf-8")
            raise EngineError(f"Could not remove the private engine: {exc}") from exc
        removed.append(str(path.absolute()))
        if event:
            event({"type": "progress", "value": 100 * (index + 1) / len(generations),
                   "message": "Removing private engine files…"})
    return {"removed": removed}
