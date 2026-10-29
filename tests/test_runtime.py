import configparser
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
import zipfile

from support import ROOT
from samstars_qgis.runtime import client, manager, worker


class ProcessTests(unittest.TestCase):
    def test_python_and_geospatial_overrides_are_isolated(self):
        with patch.dict(os.environ, {
            "PYTHONPATH": "/qgis/python", "PYTHONHOME": "/qgis",
            "QT_PLUGIN_PATH": "/qgis/qt", "GDAL_DATA": "/qgis/gdal",
            "PROJ_LIB": "/qgis/proj", "LD_LIBRARY_PATH": "/qgis/lib",
            "HTTPS_PROXY": "https://proxy.example", "SSL_CERT_FILE": "/test/ca.pem",
        }):
            env = client.sanitized_environment(sys.executable)
        for name in ("PYTHONPATH", "PYTHONHOME", "QT_PLUGIN_PATH", "LD_LIBRARY_PATH"):
            self.assertNotIn(name, env)
        self.assertNotEqual(env.get("GDAL_DATA"), "/qgis/gdal")
        self.assertNotEqual(env.get("PROJ_LIB"), "/qgis/proj")
        self.assertEqual(env["HTTPS_PROXY"], "https://proxy.example")
        self.assertEqual(env["SSL_CERT_FILE"], "/test/ca.pem")

    def test_drains_stdout_and_stderr_and_preserves_arguments(self):
        lines = []
        argument = "a path with spaces; $value"
        code, output = client.run_process(
            [sys.executable, "-I", "-c", "import sys; print(sys.argv[1]); print('error stream', file=sys.stderr)", argument],
            python=sys.executable, event=lines.append,
        )
        self.assertEqual(code, 0)
        self.assertIn(argument, lines)
        self.assertIn("error stream", output)

    def test_cancellation_before_launch(self):
        with patch.object(subprocess, "Popen") as popen:
            with self.assertRaises(client.EngineCanceled):
                client.run_process([sys.executable], cancel=lambda: True)
            popen.assert_not_called()

    def test_cancellation_while_running(self):
        canceled = threading.Event()
        timer = threading.Timer(0.25, canceled.set)
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(client.EngineCanceled):
                client.run_process([sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                                   python=sys.executable, cancel=canceled.is_set)
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 6)

    def test_timeout(self):
        with self.assertRaisesRegex(client.EngineError, "timed out"):
            client.run_process([sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                               python=sys.executable, timeout=0.2)

    @unittest.skipIf(os.name == "nt", "POSIX process group check")
    def test_cancels_child_after_group_leader_exits(self):
        # A child inherits the stdout pipe and ignores TERM. The leader exits.
        code = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
            "print('started', flush=True)"
        )
        started = time.monotonic()
        with self.assertRaisesRegex(client.EngineError, "timed out"):
            client.run_process([sys.executable, "-I", "-c", code], python=sys.executable, timeout=0.4)
        self.assertLess(time.monotonic() - started, 6)

    def test_worker_rejects_unknown_operation(self):
        with self.assertRaisesRegex(client.EngineError, "Unknown operation"):
            client.call_with_python(sys.executable, "unsupported", {})

    def test_worker_writes_structured_protocol_error(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            result = Path(directory) / "result.json"
            request.write_text('{"protocol": 999, "payload": {}}')
            process = subprocess.run([sys.executable, "-I", str(ROOT / "runtime/worker.py"),
                                      "--request", str(request), "--result", str(result)], capture_output=True)
            self.assertEqual(process.returncode, 1)
            document = json.loads(result.read_text())
            self.assertFalse(document["ok"])
            self.assertIn("protocol", document["error"]["message"])


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_existing_outputs_are_not_reused(self):
        output = self.root / "output"
        output.mkdir()
        (output / "manifest.json").write_text("original")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            worker.validate_output(output, [])
        self.assertEqual((output / "manifest.json").read_text(), "original")

    def test_model_and_input_containment_is_rejected(self):
        model = self.root / "model"
        model.mkdir()
        image = self.root / "image.tif"
        image.touch()
        for output in (model, model / "run", self.root, image):
            with self.subTest(output=output), self.assertRaises(ValueError):
                worker.validate_output(output, [model, image])

    def test_symlink_model_alias_is_rejected(self):
        model = self.root / "model"
        model.mkdir()
        link = self.root / "alias"
        try:
            link.symlink_to(model, target_is_directory=True)
        except OSError:
            self.skipTest("Symlinks unavailable")
        with self.assertRaises(ValueError):
            worker.validate_output(link, [model])

    def test_output_claim_is_exclusive_and_cleans_up(self):
        output = self.root / "new"
        with worker.claim_output(output):
            with self.assertRaisesRegex(ValueError, "Another SAMStars run"):
                with worker.claim_output(output):
                    self.fail("Concurrent job acquired the same output")
        self.assertFalse((output / ".samstars-running").exists())

    def test_output_claim_rechecks_after_validation(self):
        output = worker.validate_output(self.root / "new", [])
        output.mkdir()
        (output / "late-result").touch()
        with self.assertRaisesRegex(ValueError, "no longer empty"):
            with worker.claim_output(output):
                pass
        self.assertTrue((output / "late-result").is_file())
        self.assertFalse((output / ".samstars-running").exists())


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = {key: str(self.root / f"{key}.tif") for key in ("img", "chm", "den")}
        self.payload.update(out=str(self.root / "output"), storage="temp", device="cpu")
        self.samstars = types.ModuleType("samstars")
        self.samstars.SamStars = types.SimpleNamespace(load=lambda path: types.SimpleNamespace(defaults={"rgb_bands": [3, 2, 1]}, manifest={"bundle_version": 1}))
        self.torch = types.ModuleType("torch")
        self.torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        self.torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        for patcher in (
            patch.dict(sys.modules, {"samstars": self.samstars, "torch": self.torch}),
            patch.object(worker.importlib.metadata, "version", return_value="0.2.0"),
            patch.object(worker, "validate_rasters"), patch.object(worker, "emit"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_segment_preserves_model_band_defaults(self):
        calls = []

        def segment(**kwargs):
            calls.append(kwargs)
            crowns = kwargs["out"] / "crowns" / "crowns.gpkg"
            crowns.parent.mkdir()
            crowns.touch()
            return types.SimpleNamespace(out=kwargs["out"], crowns=crowns)

        self.samstars.segment = segment
        result = worker.run_job("segment", {**self.payload, "model": str(self.root / "model")})
        self.assertTrue(Path(result["CROWNS"]).is_file())
        self.assertNotIn("rgb_bands", calls[0])
        self.assertTrue(calls[0]["stardist_cpu"])
        self.assertIsInstance(calls[0]["model"], Path)
        worker.validate_rasters.assert_called_once()
        self.assertEqual(worker.validate_rasters.call_args.args[1], {"rgb_bands": [3, 2, 1]})

    def test_train_maps_bands_epochs_and_paths(self):
        calls = []

        def train(**kwargs):
            calls.append(kwargs)
            (kwargs["out"] / "manifest.json").write_text("{}")
            return types.SimpleNamespace(root=kwargs["out"])

        self.samstars.train = train
        result = worker.run_job("train", {
            **self.payload, "crowns": str(self.root / "crowns.gpkg"),
            "sam_base_ckpt": str(self.root / "sam.pth"), "rgb_bands": [3, 2, 1],
            "sam_epochs": 2, "stardist_epochs": 3,
        })
        self.assertEqual(result["MODEL"], self.payload["out"])
        self.assertEqual(calls[0]["rgb_bands"], (3, 2, 1))
        self.assertEqual(calls[0]["sam_epochs"], 2)
        self.assertEqual(calls[0]["stardist_epochs"], 3)
        self.assertIsInstance(calls[0]["crowns"], Path)

    def test_no_crown_output_is_failure(self):
        self.samstars.segment = lambda **kwargs: types.SimpleNamespace(out=kwargs["out"], crowns=kwargs["out"] / "missing.gpkg")
        with self.assertRaisesRegex(RuntimeError, "no crown GeoPackage"):
            worker.run_job("segment", {**self.payload, "model": str(self.root / "model")})

    def test_gpu_unavailable_is_actionable(self):
        for device in ("cuda", "mps"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "unavailable"):
                worker.run_job("segment", {**self.payload, "model": str(self.root / "model"), "device": device})


class InstallerTests(unittest.TestCase):
    def test_success_preserves_interpreter_path_for_setup(self):
        with patch.object(manager, "check_python", return_value=sys.executable), \
             patch.object(manager, "call_with_python", return_value={"ready": True, "checks": [], "python": "3.12.0"}):
            result = manager.check_engine(sys.executable)
        self.assertEqual(result["python"], sys.executable)

    def test_rejects_unsupported_python_without_installing(self):
        with patch.object(manager, "run_process", return_value=(0, "[3, 13]")):
            with self.assertRaisesRegex(client.EngineError, "3.10, 3.11, or 3.12"):
                manager.check_python(sys.executable)

    def test_failed_install_preserves_old_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "existing"
            old.mkdir()
            (old / "keep").write_text("working engine")
            with patch.object(manager.platform, "system", return_value="Linux"), \
                 patch.object(manager.platform, "machine", return_value="x86_64"), \
                 patch.object(manager, "check_python", return_value=sys.executable), \
                 patch.object(manager, "run_process", return_value=(1, "dependency resolution failed")):
                with self.assertRaisesRegex(client.EngineError, "dependency resolution failed"):
                    manager.install_engine(sys.executable, root)
            self.assertEqual(list(root.iterdir()), [old])
            self.assertEqual((old / "keep").read_text(), "working engine")


class PackagingTests(unittest.TestCase):
    def test_qgis_zip_is_self_contained(self):
        spec = importlib.util.spec_from_file_location("package_plugin", ROOT / "scripts/package_plugin.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            archive = module.package(Path(directory) / "plugin.zip")
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertTrue(all(name.startswith("samstars_qgis/") for name in names))
                for name in ("__init__.py", "metadata.txt", "runtime/worker.py", "docs/index.html", "NOTICE", "LICENSE"):
                    self.assertIn("samstars_qgis/" + name, names)
                self.assertFalse(any("__pycache__" in name or "/tests/" in name for name in names))
                metadata = configparser.ConfigParser()
                metadata.read_string(zipped.read("samstars_qgis/metadata.txt").decode())
                self.assertEqual(metadata["general"]["qgisMinimumVersion"], "4.0")


if __name__ == "__main__":
    unittest.main()
