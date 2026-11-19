"""Standalone worker: never import QGIS or modify QGIS's Python environment."""

import argparse
from contextlib import contextmanager
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import traceback

PROTOCOL = 1
SAMSTARS_VERSION = "0.2.0"
SAMSTARS_VERSIONS = {SAMSTARS_VERSION: "Stable", "0.3.0a1": "Experimental"}
PROFILE_FILE = ".samstars-runtime.json"
_DLL_HANDLES = []


def engine_profile():
    path = Path(sys.prefix) / PROFILE_FILE
    if not path.is_file():
        return {}
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
        if profile["device"] not in ("cpu", "cuda", "mps") or profile["refinement"] != "cpu":
            raise ValueError("Unsupported profile")
        return profile
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("Private engine settings are invalid. Repair the engine in SAMStars Setup.") from exc


def emit(message, value=None):
    event = {"type": "log", "message": str(message)}
    if value is not None:
        event.update(type="progress", value=value)
    print(json.dumps(event), flush=True)


def validate_version(version, expected=None):
    if expected is not None and expected not in SAMSTARS_VERSIONS:
        raise ValueError(f"Unsupported engine version: {expected}.")
    if version not in SAMSTARS_VERSIONS:
        raise RuntimeError(
            f"Found samstars {version}; supported engine versions are "
            + ", ".join(SAMSTARS_VERSIONS) + ". Choose a supported version in SAMStars Setup."
        )
    if expected is not None and version != expected:
        raise RuntimeError(f"Expected samstars=={expected}; found {version}.")
    return version


def check_dependencies(expected_version=None):
    checks = []
    modules = {}
    samstars_version = ""
    for distribution, module in (
        ("samstars", "samstars"), ("torch", "torch"),
        ("torchvision", "torchvision"), ("tensorflow", "tensorflow"),
        ("stardist", "stardist.models"), ("csbdeep", "csbdeep"),
        ("rasterio", "rasterio"), ("geopandas", "geopandas"),
        ("pyogrio", "pyogrio"), ("numpy", "numpy"),
        ("scipy", "scipy"), ("scikit-image", "skimage"),
        ("scikit-learn", "sklearn"), ("segment-anything", "segment_anything"),
    ):
        try:
            version = importlib.metadata.version(distribution)
            if distribution == "samstars":
                samstars_version = version
                validate_version(version, expected_version)
            modules[distribution] = importlib.import_module(module)
            if distribution == "tensorflow":
                # A CPU validation must also stay on CPU with a Metal plugin.
                modules[distribution].config.set_visible_devices([], "GPU")
            if distribution == "samstars":
                getattr(modules[distribution], "train")
                getattr(modules[distribution], "segment")
            checks.append({"name": distribution, "version": version, "ok": True, "error": ""})
            emit(f"OK: {distribution} {version}")
        except Exception as exc:
            checks.append({"name": distribution, "version": "", "ok": False, "error": str(exc)})
            emit(f"FAIL: {distribution}: {exc}")
    if all(item["ok"] for item in checks):
        labels = modules["numpy"].zeros((16, 16), dtype="uint16")
        labels[4:12, 4:12] = 1
        try:
            torch = modules["torch"]
            tf = modules["tensorflow"]
            if (torch.ones((2, 2)) @ torch.ones((2, 2))).sum().item() != 8:
                raise RuntimeError("PyTorch CPU operation failed.")
            with tf.device("/CPU:0"):
                product = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
                if float(tf.reduce_sum(product)) != 8:
                    raise RuntimeError("TensorFlow CPU operation failed.")
            # Exercise the compiled StarDist extension, not only its Python imports.
            from stardist.geometry import star_dist
            star_dist(labels, n_rays=8)
            checks.append({"name": "CPU operations", "version": "", "ok": True, "error": ""})
        except Exception as exc:
            checks.append({"name": "CPU operations", "version": "", "ok": False, "error": str(exc)})
        try:
            from rasterio.io import MemoryFile
            from rasterio.transform import from_origin
            from rasterio.warp import transform_bounds
            from shapely.geometry import box

            with MemoryFile() as memory:
                with memory.open(driver="GTiff", width=16, height=16, count=1,
                                 dtype="uint16", crs="EPSG:32604",
                                 transform=from_origin(500000, 2200016, 1, 1)) as raster:
                    raster.write(labels, 1)
                    transform_bounds(raster.crs, "EPSG:4326", *raster.bounds)
            frame = modules["geopandas"].GeoDataFrame(
                {"tree_id": [1]}, geometry=[box(500000, 2200000, 500001, 2200001)], crs="EPSG:32604",
            )
            with tempfile.TemporaryDirectory(prefix="samstars-check-") as directory:
                path = Path(directory) / "crowns.gpkg"
                frame.to_file(path, driver="GPKG", engine="pyogrio")
                if len(modules["geopandas"].read_file(path, engine="pyogrio")) != 1:
                    raise RuntimeError("GeoPackage round trip failed.")
            checks.append({"name": "Geospatial operations", "version": "", "ok": True, "error": ""})
        except Exception as exc:
            checks.append({"name": "Geospatial operations", "version": "", "ok": False, "error": str(exc)})
    return {"ready": all(item["ok"] for item in checks), "checks": checks,
            "python_version": sys.version.split()[0], "samstars_version": samstars_version}


def _torch_gpu_operations(torch, device):
    """Small forward/backward kernels; availability alone is not validation."""
    values = torch.ones((4, 4), device=device, requires_grad=True)
    product = values @ values
    if product.device.type != device or not torch.allclose(product, torch.full_like(product, 4)):
        raise RuntimeError("GPU matrix multiplication returned an unexpected device or result.")
    product.sum().backward()
    if not torch.allclose(values.grad, torch.full_like(values, 8)):
        raise RuntimeError("GPU matrix gradients are incorrect.")
    pixels = torch.ones((1, 1, 8, 8), device=device, requires_grad=True)
    weights = torch.ones((1, 1, 3, 3), device=device, requires_grad=True)
    convolution = torch.nn.functional.conv2d(pixels, weights)
    if convolution.device.type != device or not torch.allclose(convolution, torch.full_like(convolution, 9)):
        raise RuntimeError("GPU convolution failed.")
    convolution.sum().backward()
    if not torch.isfinite(pixels.grad).all().item() or not torch.isfinite(weights.grad).all().item():
        raise RuntimeError("GPU convolution gradients are not finite.")
    if device == "cuda":
        torch.cuda.synchronize()
    else:
        torch.mps.synchronize()


def check_gpu(device):
    if device not in ("cuda", "mps"):
        raise ValueError("GPU checks require cuda or mps.")
    check = {"name": "PyTorch GPU computation", "version": "", "ok": False, "error": ""}
    hardware = ""
    try:
        import torch
        import torchvision

        check["version"] = f"torch {torch.__version__}; torchvision {torchvision.__version__}"
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver, installed PyTorch build, and CUDA_VISIBLE_DEVICES.")
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            hardware = f"{properties.name}; {properties.total_memory / 1024 ** 3:.1f} GiB; CUDA {torch.version.cuda}"
        else:
            if not torch.backends.mps.is_available():
                raise RuntimeError("Apple MPS is unavailable. Use a native Apple Silicon Python and a supported macOS version.")
            hardware = "Apple MPS"
        _torch_gpu_operations(torch, device)
        check["ok"] = True
        emit(f"GPU computation and gradients passed: {hardware}")
    except Exception as exc:
        check["error"] = str(exc)
        emit(f"GPU check failed: {exc}")
    return {
        "ready": check["ok"], "checks": [check], "device": device,
        "hardware": hardware, "refinement": engine_profile().get("refinement", "not tested"),
    }


def _check_output_files(output, filenames):
    for name in filenames:
        path = output / name
        if path.exists() or path.is_symlink():
            raise ValueError(f"Output file already exists: {path}. Choose a different output folder or move the existing file.")


def validate_output(path, inputs, *, filenames=None):
    """Protect whole run folders, or just the named files for a standalone step."""
    output = Path(path).expanduser().resolve()
    for value in inputs:
        source = Path(value).resolve()
        if output == source or (filenames is None and output in source.parents) or (source.is_dir() and source in output.parents):
            raise ValueError("Output must be separate from input files and the model bundle.")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output must be a folder: {output}.")
    if filenames is not None:
        _check_output_files(output, filenames)
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output folder. Existing runs are never overwritten.")
    return output


@contextmanager
def claim_output(output, *, filenames=None):
    """Lock the destination and recheck its files immediately before writing."""
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".samstars-running"
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError("Another SAMStars run is using this output folder. Wait for it to finish or choose a different folder.") from exc
    try:
        if filenames is not None:
            _check_output_files(output, filenames)
        elif any(path != marker for path in output.iterdir()):
            raise ValueError("Output folder is no longer empty. Choose a new folder.")
        yield
    finally:
        marker.unlink(missing_ok=True)


def validate_rasters(payload, defaults=None):
    import rasterio
    from rasterio.warp import transform_bounds

    defaults = defaults or {}
    with rasterio.open(payload["img"]) as image:
        if not image.crs or not image.crs.is_projected:
            raise ValueError("Imagery needs a projected CRS in metres. Reproject it before running SAMStars.")
        if not math.isclose(image.crs.linear_units_factor[1], 1.0):
            raise ValueError("The imagery CRS must use metres.")
        transform = image.transform
        if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0 or not math.isclose(transform.a, -transform.e, rel_tol=1e-6):
            raise ValueError("Imagery must be north-up with square pixels. Warp it to a square grid first.")
        bands = payload.get("rgb_bands") or defaults.get("rgb_bands") or [1, 2, 3]
        if len(bands) != 3 or len(set(bands)) != 3:
            raise ValueError("RGB bands must contain three distinct indices in red, green, blue order.")
        for name in ("nir_band", "texture_band"):
            value = payload.get(name) or defaults.get(name)
            if value is not None:
                bands = [*bands, value]
        if any(not isinstance(band, int) or not 1 <= band <= image.count for band in bands):
            raise ValueError(f"Selected image bands exceed the available {image.count} bands.")
        for key, label in (("chm", "Canopy height"), ("den", "First-return density")):
            with rasterio.open(payload[key]) as raster:
                if raster.count != 1 or not raster.crs:
                    raise ValueError(f"{label} must be a single-band raster with a defined CRS.")
                if raster.scales[0] != 1 or raster.offsets[0] != 0:
                    raise ValueError(f"Apply the {label.lower()} band scale/offset before processing.")
                if key == "chm" and raster.units[0] and raster.units[0].lower() not in ("m", "metre", "metres", "meter", "meters"):
                    raise ValueError("Canopy heights must store values in metres.")
                bounds = transform_bounds(raster.crs, image.crs, *raster.bounds)
                if bounds[0] >= image.bounds.right or bounds[2] <= image.bounds.left or bounds[1] >= image.bounds.top or bounds[3] <= image.bounds.bottom:
                    raise ValueError(f"{label} does not overlap the imagery.")


def run_job(operation, payload):
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("The SAMStars engine requires Python 3.10–3.12.")
    if operation == "check":
        return check_dependencies(payload.get("samstars_version"))
    if operation == "check_gpu":
        return check_gpu(payload.get("device"))
    step_operations = {"alignrasters", "costsurface", "generateseeds", "proposals", "refine"}
    if operation not in {"train", "segment", *step_operations}:
        raise ValueError(f"Unknown operation: {operation}")
    version = importlib.metadata.version("samstars")
    validate_version(version)
    emit(f"Using SAMStars {version}.")
    if operation in step_operations:
        if __package__:
            steps = importlib.import_module(".steps", __package__)
        else:
            # -I intentionally omits the script directory from sys.path.
            spec = importlib.util.spec_from_file_location("samstars_workflow_steps", Path(__file__).with_name("steps.py"))
            steps = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(steps)
        return steps.run_step(operation, payload, emit=emit, validate_output=validate_output, claim_output=claim_output)
    import samstars

    arguments = dict(payload)
    inputs = [arguments[key] for key in ("img", "chm", "den")]
    inputs += [arguments[key] for key in ("model", "sam_base_ckpt", "crowns") if key in arguments]
    arguments["out"] = validate_output(arguments["out"], inputs)
    defaults = {}
    if operation == "segment":
        model = samstars.SamStars.load(Path(arguments["model"]))
        defaults = model.defaults
        if model.manifest.get("bundle_version") != 1:
            raise ValueError("Unsupported model bundle version; use a SAMStars bundle with bundle_version 1.")
    validate_rasters(arguments, defaults)
    for name in ("img", "chm", "den", "model", "sam_base_ckpt", "crowns"):
        if name in arguments:
            arguments[name] = Path(arguments[name])
    if arguments.get("rgb_bands"):
        arguments["rgb_bands"] = tuple(arguments["rgb_bands"])
    device = arguments.get("device", "cpu")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Device must be cpu, cuda, or mps.")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this engine. Select CPU or configure a CUDA environment in setup.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this engine. Select CPU or use Apple Silicon with an MPS-enabled PyTorch.")
    arguments["stardist_cpu"] = device == "cpu" or engine_profile().get("refinement") == "cpu"
    if device != "cpu" and arguments["stardist_cpu"]:
        emit("SAM proposals use the selected GPU; StarDist refinement uses CPU in this private engine.")
    emit("Running SAMStars. Stage logs follow; large scenes can take a long time.", 5)
    with claim_output(arguments["out"]):
        if operation == "train":
            result = samstars.train(**arguments)
            manifest = Path(result.root) / "manifest.json"
            if not manifest.is_file():
                raise RuntimeError("Training completed without a model manifest.")
            output = {"OUTPUT": str(result.root), "MODEL": str(result.root)}
        else:
            result = samstars.segment(**arguments)
            if not Path(result.crowns).is_file():
                raise RuntimeError("Segmentation produced no crown GeoPackage. Check input coverage and the stage logs.")
            output = {"OUTPUT": str(result.out), "CROWNS": str(result.crowns)}
    emit("SAMStars finished.", 100)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for relative in ("Library/bin", "DLLs"):
            directory = Path(sys.prefix) / relative
            if directory.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))

    def terminated(signum, frame):
        raise InterruptedError("SAMStars processing canceled.")

    signal.signal(signal.SIGTERM, terminated)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
            raise ValueError("Unsupported worker protocol.")
        if not isinstance(request.get("payload"), dict):
            raise ValueError("Worker payload must be an object.")
        # Disable accelerators before TensorFlow/PyTorch are imported for CPU jobs.
        if request.get("operation") == "check" or request["payload"].get("device", "cpu") == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        result = run_job(request.get("operation"), request["payload"])
        document = {"protocol": PROTOCOL, "ok": True, "result": result}
        code = 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        document = {"protocol": PROTOCOL, "ok": False, "error": {"message": str(exc)}}
        code = 1
    path = Path(args.result)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
