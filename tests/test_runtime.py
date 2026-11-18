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
from samstars_qgis.runtime import accelerators, client, manager, worker


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

    def test_gpu_probe_preserves_device_visibility_but_cpu_check_hides_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            request, result = Path(directory) / "request.json", Path(directory) / "result.json"
            for operation, payload, expected in (("check_gpu", {"device": "cuda"}, "2"), ("check", {}, "-1")):
                request.write_text(json.dumps({"protocol": 1, "operation": operation, "payload": payload}))
                visible = []
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}), \
                     patch.object(worker.signal, "signal"), \
                     patch.object(worker, "run_job", side_effect=lambda *args: visible.append(os.environ["CUDA_VISIBLE_DEVICES"]) or {}):
                    self.assertEqual(worker.main(["--request", str(request), "--result", str(result)]), 0)
                self.assertEqual(visible, [expected])


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
        for filenames in (None, ("cost.tif", "cost.workflow.json")):
            with self.subTest(filenames=filenames), worker.claim_output(output, filenames=filenames):
                with self.assertRaisesRegex(ValueError, "Another SAMStars run"):
                    with worker.claim_output(output, filenames=filenames):
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

    def test_named_outputs_allow_inputs_and_other_results_in_the_folder(self):
        image = self.root / "image.tif"
        image.write_bytes(b"original input")
        record = self.root / "workflow.json"
        record.write_bytes(b"previous step")
        files = ("cost.tif", "cost.workflow.json")
        output = worker.validate_output(self.root, [image], filenames=files)
        with worker.claim_output(output, filenames=files):
            (output / "cost.tif").write_bytes(b"result")
        self.assertEqual(image.read_bytes(), b"original input")
        self.assertEqual(record.read_bytes(), b"previous step")
        self.assertFalse((output / ".samstars-running").exists())

    def test_named_output_conflicts_are_rechecked_under_the_lock(self):
        for name in ("cost.tif", "cost.workflow.json"):
            with self.subTest(name=name):
                files = ("cost.tif", "cost.workflow.json")
                output = worker.validate_output(self.root, [], filenames=files)
                path = output / name
                path.write_bytes(b"created after validation")
                with self.assertRaisesRegex(ValueError, name):
                    with worker.claim_output(output, filenames=files):
                        self.fail("Overwrote a file created after validation")
                with self.assertRaisesRegex(ValueError, name):
                    worker.validate_output(output, [], filenames=files)
                self.assertEqual(path.read_bytes(), b"created after validation")
                self.assertFalse((output / ".samstars-running").exists())
                path.unlink()

    def test_named_outputs_do_not_replace_input_files(self):
        image = self.root / "cost.tif"
        image.write_bytes(b"input named like the result")
        with self.assertRaisesRegex(ValueError, "cost.tif"):
            worker.validate_output(self.root, [image], filenames=("cost.tif", "cost.workflow.json"))
        self.assertEqual(image.read_bytes(), b"input named like the result")

    def test_named_outputs_protect_existing_and_broken_symlinks(self):
        target = self.root / "input.tif"
        link = self.root / "cost.tif"
        for exists in (False, True):
            with self.subTest(target_exists=exists):
                if exists:
                    target.write_bytes(b"original")
                try:
                    link.symlink_to(target)
                except OSError:
                    self.skipTest("Symlinks unavailable")
                with self.assertRaisesRegex(ValueError, "cost.tif"):
                    worker.validate_output(self.root, [], filenames=("cost.tif", "cost.workflow.json"))
                self.assertTrue(link.is_symlink())
                if exists:
                    self.assertEqual(target.read_bytes(), b"original")
                else:
                    self.assertFalse(target.exists())
                link.unlink()

    def test_named_outputs_still_protect_model_directories(self):
        model = self.root / "model"
        model.mkdir()
        for output in (model, model / "run"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "model bundle"):
                worker.validate_output(output, [model], filenames=("cost.tif", "cost.workflow.json"))


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

    def test_managed_gpu_keeps_refinement_on_cpu(self):
        self.torch.cuda.is_available = lambda: True
        calls = []

        def train(**kwargs):
            calls.append(kwargs)
            (kwargs["out"] / "manifest.json").write_text("{}")
            return types.SimpleNamespace(root=kwargs["out"])

        self.samstars.train = train
        with patch.object(worker, "engine_profile", return_value={"device": "cuda", "refinement": "cpu"}):
            worker.run_job("train", {
                **self.payload, "device": "cuda", "crowns": str(self.root / "crowns.gpkg"),
                "sam_base_ckpt": str(self.root / "sam.pth"),
            })
        self.assertEqual(calls[0]["device"], "cuda")
        self.assertTrue(calls[0]["stardist_cpu"])


class AcceleratorTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.object(accelerators.platform, "system", return_value="Linux"),
            patch.object(accelerators.platform, "machine", return_value="x86_64"),
            patch.object(accelerators.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.dict(os.environ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_cuda_profile_chooses_runtime_for_card_architecture(self):
        for capability, index in (("8.6", "cu126"), ("12.0", "cu128")):
            with patch.object(accelerators, "run_process", return_value=(0, f"NVIDIA test GPU, 580.100.00, {capability}\n")):
                profile = accelerators.installation_profile("cuda")
            self.assertTrue(profile["index"].endswith("/" + index))
            self.assertEqual(profile["refinement"], "cpu")

    def test_driver_missing_old_hidden_or_unreadable_rejected(self):
        with patch.object(accelerators.shutil, "which", return_value=None):
            with self.assertRaisesRegex(client.EngineError, "driver tools"):
                accelerators.installation_profile("cuda")
        for output in ("NVIDIA GPU, 535.104.05, 8.6", "NVIDIA GPU, 560.35.05, 12.0", "N/A"):
            with patch.object(accelerators, "run_process", return_value=(0, output)), self.assertRaises(client.EngineError):
                accelerators.installation_profile("cuda")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}), \
             patch.object(accelerators, "run_process") as runner:
            with self.assertRaisesRegex(client.EngineError, "hides the GPU"):
                accelerators.installation_profile("cuda")
            runner.assert_not_called()

    def test_platform_gate_does_not_launch_programs(self):
        for system, machine, device in (
            ("Windows", "AMD64", "cuda"), ("Linux", "aarch64", "cuda"),
            ("Darwin", "x86_64", "mps"), ("Linux", "x86_64", "mps"),
        ):
            with patch.object(accelerators.platform, "system", return_value=system), \
                 patch.object(accelerators.platform, "machine", return_value=machine), \
                 patch.object(accelerators, "run_process") as runner:
                with self.assertRaises(client.EngineError):
                    accelerators.installation_profile(device)
                runner.assert_not_called()

    def test_apple_requires_native_supported_mac(self):
        with patch.object(accelerators.platform, "system", return_value="Darwin"), \
             patch.object(accelerators.platform, "machine", return_value="arm64"), \
             patch.object(accelerators.platform, "mac_ver", return_value=("14.0", (), "")):
            self.assertEqual(accelerators.installation_profile("mps")["device"], "mps")
            with patch.object(accelerators, "run_process", return_value=(0, '["Darwin", "x86_64"]')):
                with self.assertRaisesRegex(client.EngineError, "native Darwin arm64"):
                    accelerators.check_interpreter_platform(sys.executable)
            with patch.object(accelerators.platform, "mac_ver", return_value=("13.0", (), "")):
                self.assertIn("macOS 14", accelerators.platform_error("mps"))

    def test_gpu_availability_does_not_pass_failed_computation(self):
        torch = types.SimpleNamespace(
            __version__="test", version=types.SimpleNamespace(cuda="12.6"),
            cuda=types.SimpleNamespace(
                is_available=lambda: True, current_device=lambda: 0,
                get_device_properties=lambda index: types.SimpleNamespace(name="Test GPU", total_memory=1024 ** 3),
            ),
        )
        with patch.dict(sys.modules, {"torch": torch, "torchvision": types.SimpleNamespace(__version__="test")}), \
             patch.object(worker, "_torch_gpu_operations", side_effect=RuntimeError("no kernel image available")), \
             patch.object(worker, "engine_profile", return_value={}), patch.object(worker, "emit"):
            result = worker.check_gpu("cuda")
        self.assertFalse(result["ready"])
        self.assertIn("no kernel image", result["checks"][0]["error"])

    def test_invalid_private_profile_requires_repair(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(worker.sys, "prefix", directory):
            (Path(directory) / worker.PROFILE_FILE).write_text('{"device":"cuda","refinement":"automatic"}')
            with self.assertRaisesRegex(RuntimeError, "Repair the engine"):
                worker.engine_profile()


class InstallerTests(unittest.TestCase):
    def _generation(self, root, suffix="a"):
        path = root / ("engine-" + suffix * 32)
        path.mkdir(parents=True)
        (path / manager.MARKER).write_text(json.dumps(manager.OWNER))
        return path

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
            self.assertEqual([path for path in root.iterdir() if path.name != ".setup.lock"], [old])
            self.assertEqual((old / "keep").read_text(), "working engine")

    def test_install_detects_python_and_keeps_cpu_dependency_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []
            with patch.object(manager, "supported_platform", return_value=True), \
                 patch.object(manager.platform, "system", return_value="Linux"), \
                 patch.object(manager, "suggested_python", return_value=sys.executable), \
                 patch.object(manager, "check_python", return_value=sys.executable), \
                 patch.object(manager, "run_process", return_value=(0, "")) as runner, \
                 patch.object(manager, "check_engine", side_effect=lambda python, **kwargs: {"python": python, "ready": True}):
                result = manager.install_engine(root=root, event=events.append)
            commands = [call.args[0] for call in runner.call_args_list]
            self.assertEqual(commands[0][:4], [sys.executable, "-I", "-m", "venv"])
            self.assertIn("https://download.pytorch.org/whl/cpu", commands[1])
            self.assertIn(f"samstars=={worker.SAMSTARS_VERSION}", commands[2])
            self.assertFalse(any("and-cuda" in value or "tensorflow-metal" in value for cmd in commands for value in cmd))
            self.assertTrue(manager._owned_engine(Path(result["python"]).parent.parent))
            self.assertEqual([event["value"] for event in events if event["type"] == "progress"], [5, 15, 45, 85, 100])

    def test_mac_install_does_not_request_cuda_or_metal_extras(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(manager, "supported_platform", return_value=True), \
                 patch.object(manager.platform, "system", return_value="Darwin"), \
                 patch.object(manager, "check_python", return_value=sys.executable), \
                 patch.object(manager, "run_process", return_value=(0, "")) as runner, \
                 patch.object(manager, "check_engine", return_value={"python": sys.executable, "ready": True}):
                manager.install_engine(sys.executable, directory)
            commands = [call.args[0] for call in runner.call_args_list]
            self.assertEqual(len(commands), 3)
            self.assertIn(f"samstars=={worker.SAMSTARS_VERSION}", commands[1])
            self.assertFalse(any("and-cuda" in value or "metal" in value or "/whl/cpu" in value for cmd in commands for value in cmd))

    def test_gpu_install_is_constrained_and_checked_before_returning(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = {"device": "cuda", "refinement": "cpu", "index": "https://download.pytorch.org/whl/cu126", "hardware": "Test GPU"}
            with patch.object(manager, "check_python", return_value=sys.executable), \
                 patch.object(accelerators, "installation_profile", return_value=profile), \
                 patch.object(accelerators, "check_interpreter_platform"), \
                 patch.object(manager, "run_process", return_value=(0, "")) as runner, \
                 patch.object(manager, "check_engine", side_effect=lambda python, **kwargs: {"python": python, "ready": True}), \
                 patch.object(manager, "check_gpu", return_value={"ready": True}) as gpu:
                result = manager.install_engine(sys.executable, directory, accelerator="cuda")
            commands = [call.args[0] for call in runner.call_args_list]
            self.assertIn("torch==2.10.0+cu126", commands[1])
            self.assertIn("--constraint", commands[2])
            self.assertIn("--only-binary=:all:", commands[2])
            self.assertEqual(commands[-1][-1], "check")
            self.assertFalse(any("and-cuda" in value or "tensorflow-metal" in value for cmd in commands for value in cmd))
            self.assertEqual(result["accelerator"], "cuda")
            gpu.assert_called_once()
            self.assertEqual(gpu.call_args.args, (result["python"], "cuda"))
            self.assertEqual(json.loads((Path(result["python"]).parent.parent / worker.PROFILE_FILE).read_text()), profile)

    def test_gpu_probe_failure_or_cancellation_preserves_previous_engine(self):
        for error in (client.EngineError("GPU kernel failed"), client.EngineCanceled("Canceled")):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                old = self._generation(root)
                (old / "keep").write_text("previous engine")
                with patch.object(manager, "check_python", return_value=sys.executable), \
                     patch.object(accelerators, "installation_profile", return_value={"device": "cuda", "refinement": "cpu", "index": "https://download.pytorch.org/whl/cu126", "hardware": "GPU"}), \
                     patch.object(accelerators, "check_interpreter_platform"), \
                     patch.object(manager, "run_process", return_value=(0, "")), \
                     patch.object(manager, "check_engine", return_value={"ready": True}), \
                     patch.object(manager, "check_gpu", side_effect=error):
                    with self.assertRaises(type(error)):
                        manager.install_engine(sys.executable, root, accelerator="cuda")
                self.assertEqual(manager.owned_engines(root), [old])
                self.assertEqual((old / "keep").read_text(), "previous engine")

    def test_gpu_preflight_failure_does_not_create_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "engines"
            with patch.object(manager, "check_python", return_value=sys.executable), \
                 patch.object(accelerators, "installation_profile", side_effect=client.EngineError("Driver required")), \
                 patch.object(manager, "run_process") as runner:
                with self.assertRaisesRegex(client.EngineError, "Driver required"):
                    manager.install_engine(sys.executable, root, accelerator="cuda")
                runner.assert_not_called()
                self.assertFalse(root.exists())

    def test_remove_preserves_unmarked_environments_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = self._generation(root)
            (generation / "installed-package").touch()
            legacy = root / ("engine-" + "b" * 32)
            legacy.mkdir()
            (legacy / "pyvenv.cfg").write_text("existing environment")
            unrelated = root / "notes.txt"
            unrelated.write_text("keep")
            result = manager.remove_engine(root)
            self.assertEqual(result["removed"], [str(generation)])
            self.assertFalse(generation.exists())
            self.assertTrue(legacy.is_dir())
            self.assertEqual(unrelated.read_text(), "keep")

    def test_remove_does_not_follow_engine_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "engines"
            root.mkdir()
            external = self._generation(Path(directory) / "external")
            link = root / external.name
            try:
                link.symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest("Symlinks unavailable")
            self.assertEqual(manager.remove_engine(root)["removed"], [])
            self.assertTrue(external.is_dir())
            self.assertTrue(link.is_symlink())

    def test_remove_canceled_before_deleting(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = self._generation(Path(directory))
            with self.assertRaises(client.EngineCanceled):
                manager.remove_engine(directory, cancel=lambda: True)
            self.assertTrue(generation.is_dir())

    def test_removal_is_blocked_during_another_setup_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            generation = self._generation(Path(directory))
            with manager._operation_lock(directory):
                with self.assertRaisesRegex(client.EngineError, "Another SAMStars setup"):
                    manager.remove_engine(directory)
            self.assertTrue(generation.is_dir())
            self.assertEqual(manager.remove_engine(directory)["removed"], [str(generation)])

    def test_status_distinguishes_managed_and_existing_environments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = self._generation(root)
            python = generation / "bin" / "python"
            python.parent.mkdir()
            python.touch()
            with patch.object(manager, "engine_root", return_value=root), \
                 patch.object(manager, "engine_python", return_value=str(python)), \
                 patch.object(manager, "supported_platform", return_value=True):
                state = manager.status()
                self.assertTrue(state["managed"])
                self.assertTrue(state["installed"])
                self.assertTrue(state["removable"])
            with patch.object(manager, "engine_root", return_value=root), \
                 patch.object(manager, "engine_python", return_value=sys.executable), \
                 patch.object(manager, "supported_platform", return_value=True):
                state = manager.status()
                self.assertFalse(state["managed"])
                self.assertTrue(state["installed"])

    def test_broken_profile_can_still_be_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = self._generation(root)
            python = generation / "bin/python"
            python.parent.mkdir()
            python.touch()
            (generation / worker.PROFILE_FILE).write_text("broken json")
            with patch.object(manager, "engine_root", return_value=root), \
                 patch.object(manager, "engine_python", return_value=str(python)), \
                 patch.object(manager, "supported_platform", return_value=True):
                state = manager.status()
            self.assertTrue(state["managed"])
            self.assertFalse(state["error"])
            self.assertIn("Repair", state["warning"])
            self.assertEqual(state["accelerator"], "cpu")


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
                for name in ("__init__.py", "metadata.txt", "runtime/worker.py", "runtime/steps.py", "algorithms/steps.py", "runtime/accelerators.py", "docs/index.html", "NOTICE", "LICENSE"):
                    self.assertIn("samstars_qgis/" + name, names)
                self.assertFalse(any("__pycache__" in name or "/tests/" in name for name in names))
                metadata = configparser.ConfigParser()
                metadata.read_string(zipped.read("samstars_qgis/metadata.txt").decode())
                self.assertEqual(metadata["general"]["qgisMinimumVersion"], "4.0")


if __name__ == "__main__":
    unittest.main()
