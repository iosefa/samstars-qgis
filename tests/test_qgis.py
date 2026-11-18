import atexit
import gc
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import MagicMock, patch

from support import ROOT

try:
    from qgis.core import (
        Qgis, QgsApplication, QgsFeature, QgsGeometry, QgsProcessingContext,
        QgsProcessingException, QgsProcessingFeatureSourceDefinition,
        QgsProcessingFeedback, QgsProject, QgsRasterLayer, QgsVectorLayer,
    )
    from osgeo import gdal, osr
    gdal.UseExceptions()
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False

_application = None
_profile = None


def application():
    global _application, _profile
    if _application is None:
        _profile = tempfile.TemporaryDirectory(prefix="samstars-qgis-profile-")
        _application = QgsApplication([], False, _profile.name)
        _application.initQgis()

        def cleanup():
            global _application
            _application.exitQgis()
            _application = None
            _profile.cleanup()

        atexit.register(cleanup)
    return _application


@unittest.skipUnless(QGIS_AVAILABLE, "Requires QGIS Python bindings")
class QgisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        application()

    def setUp(self):
        import numpy as np

        self.temp = tempfile.TemporaryDirectory(prefix="samstars-qgis-data-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = QgsProject()
        self.context = QgsProcessingContext()
        self.context.setProject(self.project)
        self.feedback = QgsProcessingFeedback()
        self.parameters = {}
        for key, count in (("IMAGE", 4), ("CHM", 1), ("DENSITY", 1)):
            path = self.root / f"{key}.tif"
            dataset = gdal.GetDriverByName("GTiff").Create(str(path), 16, 16, count, gdal.GDT_Float32)
            crs = osr.SpatialReference()
            crs.ImportFromEPSG(32604)
            dataset.SetProjection(crs.ExportToWkt())
            dataset.SetGeoTransform((500000, 1, 0, 2200016, 0, -1))
            for band in range(1, count + 1):
                dataset.GetRasterBand(band).WriteArray(np.arange(256, dtype="float32").reshape((16, 16)) + band)
            dataset = None
            layer = QgsRasterLayer(str(path), key)
            self.assertTrue(layer.isValid())
            self.project.addMapLayer(layer)
            self.parameters[key] = layer.id()
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "manifest.json").write_text("{}")
        self.parameters.update(MODEL=str(self.model), OUTPUT=str(self.root / "output"))

    def tearDown(self):
        # Destroy layer owners while QGIS and their backing files still exist.
        self.context.setProject(None)
        self.context = None
        self.project.removeAllMapLayers()
        self.project = None
        gc.collect()

    def segment(self):
        from samstars_qgis.algorithms.segment import SegmentAlgorithm
        algorithm = SegmentAlgorithm()
        algorithm.initAlgorithm()
        return algorithm

    def test_provider_registers_eight_tools_without_ml_imports(self):
        from samstars_qgis.provider import SamStarsProvider
        before = set(sys.modules)
        provider = SamStarsProvider()
        registry = QgsApplication.processingRegistry()
        self.assertTrue(registry.addProvider(provider))
        try:
            self.assertEqual({algorithm.id() for algorithm in provider.algorithms()}, {
                "samstars:segment", "samstars:train", "samstars:checkdependencies",
                "samstars:alignrasters", "samstars:costsurface", "samstars:generateseeds",
                "samstars:proposals", "samstars:refine",
            })
            for algorithm in provider.algorithms():
                self.assertTrue(Path(algorithm.helpUrl().split("#")[0].removeprefix("file://")).is_file())
                if algorithm.name() in {"alignrasters", "costsurface", "generateseeds", "proposals", "refine"}:
                    self.assertEqual(algorithm.group(), "Workflow steps")
            self.assertFalse({"torch", "tensorflow", "samstars"} & (set(sys.modules) - before))
        finally:
            registry.removeProvider(provider)

    def test_qgis3_startup_is_rejected(self):
        from samstars_qgis.samstars_plugin import SamStarsPlugin
        with patch("samstars_qgis.samstars_plugin.Qgis", types.SimpleNamespace(QGIS_VERSION_INT=39999)):
            with self.assertRaisesRegex(RuntimeError, "QGIS 4.0"):
                SamStarsPlugin(None).initProcessing()

    def test_provider_lifecycle_is_idempotent(self):
        from samstars_qgis.samstars_plugin import SamStarsPlugin
        plugin = SamStarsPlugin(None)
        with patch("samstars_qgis.samstars_plugin.Qgis", types.SimpleNamespace(QGIS_VERSION_INT=40000)):
            plugin.initProcessing()
            first = plugin.provider
            plugin.initProcessing()
            self.assertIs(plugin.provider, first)
        plugin.unload()
        plugin.unload()
        self.assertIsNone(QgsApplication.processingRegistry().providerById("samstars"))

    def test_segment_passes_model_defaults_and_loads_result(self):
        algorithm = self.segment()
        output = {"OUTPUT": self.parameters["OUTPUT"], "CROWNS": str(self.root / "output/crowns/crowns.gpkg")}
        with patch("samstars_qgis.algorithms.common.call_engine", return_value=output) as engine:
            result = algorithm.processAlgorithm(self.parameters, self.context, self.feedback)
        self.assertEqual(result, output)
        operation, payload, _ = engine.call_args.args
        self.assertEqual(operation, "segment")
        self.assertNotIn("rgb_bands", payload)
        self.assertEqual(payload["device"], "cpu")
        self.assertEqual(payload["storage"], "temp")
        self.assertEqual(payload["img"], str(self.root / "IMAGE.tif"))
        self.assertIn(output["CROWNS"], self.context.layersToLoadOnCompletion())

    def test_temporary_output_is_resolved_to_a_folder(self):
        algorithm = self.segment()
        self.parameters["OUTPUT"] = "TEMPORARY_OUTPUT"
        with patch("samstars_qgis.algorithms.common.call_engine", return_value={"OUTPUT": "out", "CROWNS": "crowns.gpkg"}) as engine:
            algorithm.processAlgorithm(self.parameters, self.context, self.feedback)
        output = engine.call_args.args[1]["out"]
        self.assertTrue(Path(output).is_absolute())
        self.assertNotIn("TEMPORARY_OUTPUT", output)

    def test_processing_runner_applies_defaults(self):
        algorithm = self.segment()
        output = {"OUTPUT": self.parameters["OUTPUT"], "CROWNS": str(self.root / "output/crowns.gpkg")}
        with patch("samstars_qgis.algorithms.common.call_engine", return_value=output) as engine:
            result, ok = algorithm.run(self.parameters, self.context, self.feedback)
        self.assertTrue(ok)
        self.assertEqual(result, output)
        self.assertEqual(engine.call_args.args[1]["device"], "cpu")

    def test_invalid_rgb_fails_before_engine(self):
        for bands in ("1,1,2", "1,2", "1,2,9", "red,green,blue"):
            with self.subTest(bands=bands), patch("samstars_qgis.algorithms.common.call_engine") as engine:
                with self.assertRaises(QgsProcessingException):
                    self.segment().processAlgorithm({**self.parameters, "RGB_BANDS": bands}, self.context, self.feedback)
                engine.assert_not_called()

    def test_existing_output_is_not_replaced(self):
        output = Path(self.parameters["OUTPUT"])
        output.mkdir()
        (output / "keep").write_text("original")
        with patch("samstars_qgis.algorithms.common.call_engine") as engine:
            with self.assertRaisesRegex(QgsProcessingException, "new or empty"):
                self.segment().processAlgorithm(self.parameters, self.context, self.feedback)
            engine.assert_not_called()
        self.assertEqual((output / "keep").read_text(), "original")

    def test_missing_bundle_fails_before_engine(self):
        (self.model / "manifest.json").unlink()
        with self.assertRaisesRegex(QgsProcessingException, "manifest.json"):
            self.segment().processAlgorithm(self.parameters, self.context, self.feedback)

    def test_training_exports_only_selected_edited_crowns(self):
        from samstars_qgis.algorithms.train import TrainAlgorithm
        crowns = QgsVectorLayer("Polygon?crs=EPSG:32604&field=tree_id:integer", "Labeled crowns", "memory")
        crowns.startEditing()
        ids = []
        for index in (1, 2):
            feature = QgsFeature(crowns.fields())
            feature["tree_id"] = index
            x = 500000 + index * 2
            feature.setGeometry(QgsGeometry.fromWkt(f"POLYGON(({x} 2200002,{x+1} 2200002,{x+1} 2200003,{x} 2200003,{x} 2200002))"))
            self.assertTrue(crowns.addFeature(feature))
            ids.append(feature.id())
        crowns.selectByIds([ids[0]])
        self.project.addMapLayer(crowns)
        checkpoint = self.root / "sam_vit_l_0b3195.pth"
        checkpoint.touch()
        algorithm = TrainAlgorithm()
        algorithm.initAlgorithm()
        parameters = {
            **self.parameters, "CROWNS": QgsProcessingFeatureSourceDefinition(crowns.id(), True),
            "SAM_CHECKPOINT": str(checkpoint), "RGB_BANDS": "3,2,1", "NIR_BAND": 4,
            "SAM_EPOCHS": 2, "STARDIST_EPOCHS": 3, "KEEP_INTERMEDIATES": True,
        }
        with patch("samstars_qgis.algorithms.common.call_engine", return_value={"OUTPUT": parameters["OUTPUT"], "MODEL": parameters["OUTPUT"]}) as engine:
            result = algorithm.processAlgorithm(parameters, self.context, self.feedback)
        payload = engine.call_args.args[1]
        exported = QgsVectorLayer(payload["crowns"], "exported", "ogr")
        self.assertTrue(exported.isValid())
        self.assertEqual(exported.featureCount(), 1)
        self.assertEqual(next(exported.getFeatures())["tree_id"], 1)
        self.assertEqual(payload["rgb_bands"], [3, 2, 1])
        self.assertEqual(payload["nir_band"], 4)
        self.assertEqual(payload["sam_epochs"], 2)
        self.assertEqual(payload["storage"], "persistent")
        self.assertEqual(result["MODEL"], parameters["OUTPUT"])
        self.assertTrue(crowns.isEditable())

    def test_canceled_algorithm_does_not_start_engine(self):
        self.feedback.cancel()
        with patch("samstars_qgis.algorithms.common.call_engine") as engine:
            with self.assertRaisesRegex(QgsProcessingException, "canceled"):
                self.segment().processAlgorithm(self.parameters, self.context, self.feedback)
            engine.assert_not_called()

    def test_preparation_steps_apply_defaults_and_load_outputs(self):
        from samstars_qgis.algorithms.steps import AlignRastersAlgorithm, CostSurfaceAlgorithm, SeedsAlgorithm
        cases = (
            (AlignRastersAlgorithm, {"REFERENCE_RASTER": "reference.tif", "ALIGNED_IMAGE": "img.tif", "ALIGNED_CHM": "chm.tif", "ALIGNED_DENSITY": "den.tif"}, ("img.tif", "chm.tif", "den.tif")),
            (CostSurfaceAlgorithm, {"COST": "cost.tif"}, ("cost.tif",)),
            (SeedsAlgorithm, {"SEEDS": "seeds.gpkg"}, ("seeds.gpkg",)),
        )
        for cls, outputs, layers in cases:
            with self.subTest(step=cls.__name__):
                algorithm = cls()
                algorithm.initAlgorithm()
                params = {**self.parameters, "COST": self.parameters["CHM"], "REFERENCE": self.parameters["CHM"]}
                with patch("samstars_qgis.algorithms.common.call_engine", return_value={"OUTPUT": params["OUTPUT"], **outputs}) as engine:
                    result, ok = algorithm.run(params, self.context, self.feedback)
                self.assertTrue(ok)
                operation, payload, _ = engine.call_args.args
                self.assertEqual(operation, algorithm.name())
                self.assertEqual(payload["chm"], str(self.root / "CHM.tif"))
                for layer in layers:
                    self.assertIn(layer, self.context.layersToLoadOnCompletion())
                if cls is CostSurfaceAlgorithm:
                    self.assertEqual(payload["weights"], [.4, .3, .3])
                    self.assertEqual(payload["rgb_bands"], [1, 2, 3])
                if cls is SeedsAlgorithm:
                    self.assertEqual(payload["seed_settings"]["d_min"], 4.5)
                    self.assertEqual(payload["seed_settings"]["min_dist_px"], 4)
                    self.assertEqual(payload["thin_distance"], .8)

    def test_cost_and_seed_dialogs_accept_existing_input_folders(self):
        from samstars_qgis.algorithms.steps import CostSurfaceAlgorithm, SeedsAlgorithm
        (self.root / "workflow.json").write_text("previous alignment settings")
        for cls, key, filename in ((CostSurfaceAlgorithm, "COST", "cost.tif"), (SeedsAlgorithm, "SEEDS", "seeds.gpkg")):
            with self.subTest(step=cls.__name__):
                algorithm = cls()
                algorithm.initAlgorithm()
                params = {**self.parameters, "COST": self.parameters["CHM"], "OUTPUT": str(self.root)}
                outputs = {"OUTPUT": str(self.root), key: str(self.root / filename)}
                with patch("samstars_qgis.algorithms.common.call_engine", return_value=outputs) as engine:
                    result, ok = algorithm.run(params, self.context, self.feedback)
                self.assertTrue(ok)
                self.assertEqual(result, outputs)
                self.assertEqual(engine.call_args.args[1]["out"], str(self.root))
                self.assertIn(outputs[key], self.context.layersToLoadOnCompletion())
        self.assertEqual((self.root / "workflow.json").read_text(), "previous alignment settings")

    def test_cost_and_seed_dialogs_name_conflicting_output_files(self):
        from samstars_qgis.algorithms.steps import CostSurfaceAlgorithm, SeedsAlgorithm
        for cls, filenames in (
            (CostSurfaceAlgorithm, ("cost.tif", "cost.workflow.json")),
            (SeedsAlgorithm, ("seeds.gpkg", "seeds.workflow.json")),
        ):
            algorithm = cls()
            algorithm.initAlgorithm()
            params = {**self.parameters, "COST": self.parameters["CHM"], "OUTPUT": str(self.root)}
            for name in filenames:
                with self.subTest(step=cls.__name__, filename=name):
                    path = self.root / name
                    path.write_bytes(b"existing result or settings")
                    with patch("samstars_qgis.algorithms.common.call_engine") as engine:
                        with self.assertRaisesRegex(QgsProcessingException, name):
                            algorithm.processAlgorithm(params, self.context, self.feedback)
                        engine.assert_not_called()
                    self.assertEqual(path.read_bytes(), b"existing result or settings")
                    path.unlink()

    def test_alignment_accepts_each_optional_input_independently(self):
        from samstars_qgis.algorithms.steps import AlignRastersAlgorithm
        for name, key, output in (("IMAGE", "img", "ALIGNED_IMAGE"), ("CHM", "chm", "ALIGNED_CHM"),
                                  ("DENSITY", "den", "ALIGNED_DENSITY"), ("INTENSITY", "intensity", "ALIGNED_INTENSITY")):
            with self.subTest(name=name):
                algorithm = AlignRastersAlgorithm()
                algorithm.initAlgorithm()
                params = {"REFERENCE": self.parameters["CHM"], "OUTPUT": self.parameters["OUTPUT"],
                          name: self.parameters.get(name, self.parameters["DENSITY"])}
                for other in ("IMAGE", "CHM", "DENSITY", "INTENSITY"):
                    params.setdefault(other, None)
                outputs = {"OUTPUT": params["OUTPUT"], "REFERENCE_RASTER": str(self.root / "CHM.tif"),
                           output: str(self.root / f"aligned_{name}.tif")}
                with patch("samstars_qgis.algorithms.common.call_engine", return_value=outputs) as engine:
                    result, ok = algorithm.run(params, self.context, self.feedback)
                self.assertTrue(ok)
                self.assertEqual(result, outputs)
                payload = engine.call_args.args[1]
                self.assertEqual(set(payload), {"reference", key, "out"})
                self.assertEqual(payload["reference"], str(self.root / "CHM.tif"))
                self.assertIn(outputs[output], self.context.layersToLoadOnCompletion())

    def test_alignment_requires_at_least_one_optional_input(self):
        from samstars_qgis.algorithms.steps import AlignRastersAlgorithm
        algorithm = AlignRastersAlgorithm()
        algorithm.initAlgorithm()
        with patch("samstars_qgis.algorithms.common.call_engine") as engine:
            with self.assertRaisesRegex(QgsProcessingException, "at least one"):
                algorithm.processAlgorithm({"REFERENCE": self.parameters["CHM"], "OUTPUT": self.parameters["OUTPUT"]},
                                           self.context, self.feedback)
            engine.assert_not_called()

    def test_alignment_reprojects_geographic_imagery_but_requires_projected_reference(self):
        from samstars_qgis.algorithms.steps import AlignRastersAlgorithm
        path = self.root / "geographic.tif"
        dataset = gdal.Translate(str(path), str(self.root / "IMAGE.tif"), outputSRS="EPSG:4326")
        dataset.SetGeoTransform((-156, .00001, 0, 20, 0, -.00001))
        dataset = None
        layer = QgsRasterLayer(str(path), "Geographic imagery")
        self.assertTrue(layer.isValid())
        self.project.addMapLayer(layer)
        algorithm = AlignRastersAlgorithm()
        algorithm.initAlgorithm()
        params = {"REFERENCE": self.parameters["CHM"], "IMAGE": layer.id(), "OUTPUT": self.parameters["OUTPUT"]}
        outputs = {"OUTPUT": params["OUTPUT"], "REFERENCE_RASTER": str(self.root / "CHM.tif"), "ALIGNED_IMAGE": "aligned.tif"}
        with patch("samstars_qgis.algorithms.common.call_engine", return_value=outputs):
            _, ok = algorithm.run(params, self.context, self.feedback)
        self.assertTrue(ok)
        with self.assertRaisesRegex(QgsProcessingException, "projected CRS"):
            algorithm.processAlgorithm({**params, "REFERENCE": layer.id()}, self.context, self.feedback)

    def test_proposals_use_selected_unsaved_seed_edits(self):
        from samstars_qgis.algorithms.steps import ProposalsAlgorithm
        seeds = QgsVectorLayer("Point?crs=EPSG:32604&field=tree_id:integer", "Seeds", "memory")
        seeds.startEditing()
        ids = []
        for index in (1, 2):
            feature = QgsFeature(seeds.fields())
            feature["tree_id"] = index
            feature.setGeometry(QgsGeometry.fromWkt(f"POINT({500000 + index} 2200003)"))
            self.assertTrue(seeds.addFeature(feature))
            ids.append(feature.id())
        moved = QgsGeometry.fromWkt("POINT(500009 2200007)")
        self.assertTrue(seeds.changeGeometry(ids[0], moved))
        seeds.selectByIds([ids[0]])
        self.project.addMapLayer(seeds)
        algorithm = ProposalsAlgorithm()
        algorithm.initAlgorithm()
        params = {**self.parameters, "COST": self.parameters["CHM"],
                  "SEEDS": QgsProcessingFeatureSourceDefinition(seeds.id(), True), "CROP_SMOOTHING": 1.5}
        outputs = {"OUTPUT": params["OUTPUT"], "PROPOSAL_RECORDS": "records.json", "PROBABILITY": "prob.tif", "CANOPY_INDEX": "index.tif"}
        with patch("samstars_qgis.algorithms.common.call_engine", return_value=outputs) as engine:
            result, ok = algorithm.run(params, self.context, self.feedback)
        self.assertTrue(ok)
        self.assertEqual(result, outputs)
        self.assertEqual(engine.call_args.args[0], "proposals")
        payload = engine.call_args.args[1]
        exported = QgsVectorLayer(payload["seeds"], "exported", "ogr")
        self.assertEqual(exported.featureCount(), 1)
        feature = next(exported.getFeatures())
        self.assertEqual(feature["tree_id"], 1)
        self.assertEqual(feature.geometry().asWkt(), moved.asWkt())
        self.assertTrue(seeds.isEditable())
        self.assertEqual(payload["device"], "cpu")
        self.assertEqual(payload["crop_smooth_sigma"], 1.5)
        self.assertNotIn("rgb_bands", payload)
        self.assertNotIn("smooth_sigma", payload)
        self.assertNotIn("den", payload)
        self.assertIn("prob.tif", self.context.layersToLoadOnCompletion())

    def test_proposals_reject_empty_seed_selection(self):
        from samstars_qgis.algorithms.steps import ProposalsAlgorithm
        layer = QgsVectorLayer("Point?crs=EPSG:32604", "Seeds", "memory")
        self.project.addMapLayer(layer)
        algorithm = ProposalsAlgorithm()
        algorithm.initAlgorithm()
        with self.assertRaisesRegex(QgsProcessingException, "at least one"):
            algorithm.export_seeds({"SEEDS": layer.id()}, self.context, self.feedback)

    def test_refinement_defaults_overrides_and_output_loading(self):
        from samstars_qgis.algorithms.steps import RefineAlgorithm
        records = self.root / "proposal_records.json"
        records.write_text("[]")
        algorithm = RefineAlgorithm()
        algorithm.initAlgorithm()
        params = {**self.parameters, "PROPOSAL_RECORDS": str(records)}
        outputs = {"OUTPUT": params["OUTPUT"], "CROWNS": "refined.gpkg"}
        with patch("samstars_qgis.algorithms.common.call_engine", return_value=outputs) as engine:
            _, ok = algorithm.run(params, self.context, self.feedback)
        self.assertTrue(ok)
        payload = engine.call_args.args[1]
        self.assertEqual(engine.call_args.args[0], "refine")
        self.assertNotIn("stardist_prob_thr", payload)
        self.assertNotIn("stardist_window_size_m", payload)
        self.assertIn(outputs["CROWNS"], self.context.layersToLoadOnCompletion())
        with self.assertRaisesRegex(QgsProcessingException, "both refinement thresholds"):
            algorithm.processAlgorithm({**params, "PROB_THRESHOLD": .4}, self.context, self.feedback)
        with patch("samstars_qgis.algorithms.common.call_engine", return_value=outputs) as engine:
            _, ok = algorithm.run({**params, "PROB_THRESHOLD": .4, "NMS_THRESHOLD": .3, "WINDOW_SIZE": 50}, self.context, self.feedback)
        self.assertTrue(ok)
        self.assertEqual(engine.call_args.args[1]["stardist_window_size_m"], 50)

    def test_dependency_report_survives_missing_engine_and_escapes_html(self):
        from samstars_qgis.algorithms.dependency_check import DependencyCheckAlgorithm
        from samstars_qgis.runtime.client import EngineError
        algorithm = DependencyCheckAlgorithm()
        algorithm.initAlgorithm()
        report = self.root / "report.html"
        with patch("samstars_qgis.algorithms.dependency_check.call_engine", side_effect=EngineError("Missing <engine>")):
            result = algorithm.processAlgorithm({"REPORT": str(report)}, self.context, self.feedback)
        self.assertFalse(result["READY"])
        self.assertIn("Missing &lt;engine&gt;", report.read_text())

    def test_setup_dialog_can_open_without_engine(self):
        from samstars_qgis.setup_dialog import SetupDialog
        dialog = SetupDialog()
        try:
            self.assertEqual(dialog.windowTitle(), "SAMStars Setup")
            self.assertFalse(dialog.cancel_button.isEnabled())
            self.assertTrue(dialog.advanced.isCollapsed())
            self.assertIn(dialog.install_button.text(), ("Install engine", "Repair engine"))
            self.assertEqual(dialog.remove_button.text(), "Remove engine")
            from qgis.PyQt.QtWidgets import QLabel
            self.assertFalse(any("0.2.0" in label.text() for label in dialog.findChildren(QLabel)))
            self.assertIsNone(dialog._thread)
            self.assertTrue(dialog.gpu_button.isEnabled())
        finally:
            dialog.close()

    def test_gpu_dialog_blocks_unsupported_install_but_can_test_existing_engine(self):
        from samstars_qgis.setup_dialog import GpuSetupDialog
        state = {"installed": True, "override": False, "error": "", "accelerator": "cpu"}
        with patch("samstars_qgis.setup_dialog.accelerators.platform_error", return_value="Unsupported platform"):
            dialog = GpuSetupDialog(state)
            try:
                dialog.device.setCurrentIndex(dialog.device.findData("cuda"))
                self.assertFalse(dialog.install_button.isEnabled())
                self.assertTrue(dialog.test_button.isEnabled())
                self.assertIn("Unsupported platform", dialog.details.text())
            finally:
                dialog.close()

    def test_gpu_dialog_explains_refinement_and_respects_environment_override(self):
        from samstars_qgis.setup_dialog import GpuSetupDialog
        from qgis.PyQt.QtWidgets import QLabel
        state = {"installed": True, "override": True, "error": "", "accelerator": "cuda"}
        dialog = GpuSetupDialog(state)
        try:
            self.assertFalse(dialog.install_button.isEnabled())
            self.assertTrue(dialog.test_button.isEnabled())
            self.assertIn("SAMSTARS_PYTHON", dialog.details.text())
            self.assertTrue(any("StarDist refinement stays on CPU" in label.text() for label in dialog.findChildren(QLabel)))
        finally:
            dialog.close()

    def test_gpu_diagnostic_success_never_changes_selected_environment(self):
        from samstars_qgis.setup_dialog import SetupDialog
        dialog = SetupDialog()
        try:
            dialog._diagnosing = True
            with patch("samstars_qgis.setup_dialog.QgsSettings") as settings:
                dialog._succeeded({"ready": True, "hardware": "GPU", "refinement": "cpu", "checks": []})
                settings.assert_not_called()
            dialog._finished()
            self.assertIn("GPU test passed", dialog.message.text())
            self.assertEqual(dialog.progress_bar.value(), 100)
        finally:
            dialog.close()

    def test_repair_preserves_gpu_profile_and_cpu_can_be_selected_explicitly(self):
        from samstars_qgis.setup_dialog import SetupDialog
        dialog = SetupDialog()
        try:
            with patch("samstars_qgis.setup_dialog.manager.status", return_value={"managed": True, "accelerator": "cuda"}), \
                 patch("samstars_qgis.setup_dialog.manager.install_engine") as installer, \
                 patch.object(dialog, "_start", side_effect=lambda operation: operation()):
                dialog._install()
                self.assertEqual(installer.call_args.kwargs["accelerator"], "cuda")
                dialog._install(accelerator="cpu")
                self.assertEqual(installer.call_args.kwargs["accelerator"], "cpu")
        finally:
            dialog.close()

    def test_gpu_button_uses_diagnostic_operation_without_installing(self):
        from samstars_qgis.setup_dialog import SetupDialog
        from qgis.PyQt.QtWidgets import QDialog
        dialog = SetupDialog()
        try:
            with patch("samstars_qgis.setup_dialog.manager.status", return_value={"python": "/tmp/selected/python"}), \
                 patch("samstars_qgis.setup_dialog.GpuSetupDialog") as gpu_dialog, \
                 patch("samstars_qgis.setup_dialog.manager.check_gpu", return_value={}) as check, \
                 patch("samstars_qgis.setup_dialog.manager.install_engine") as installer, \
                 patch.object(dialog, "_start", side_effect=lambda operation, **kwargs: operation()) as start:
                gpu_dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
                gpu_dialog.return_value.device.currentData.return_value = "cuda"
                gpu_dialog.return_value.operation = "test"
                dialog._gpu_setup()
                check.assert_called_once_with("/tmp/selected/python", "cuda")
                installer.assert_not_called()
                self.assertTrue(start.call_args.kwargs["diagnosing"])
        finally:
            dialog.close()

    def test_setup_failure_stays_visible_and_does_not_show_completion(self):
        from samstars_qgis.setup_dialog import SetupDialog
        dialog = SetupDialog()
        try:
            with patch("samstars_qgis.setup_dialog.QgsSettings") as settings:
                dialog._failed("The selected Python could not start.", False)
                dialog._finished()
                settings.assert_not_called()
            self.assertIn("could not start", dialog.message.text())
            self.assertFalse(dialog.advanced.isCollapsed())
            self.assertEqual(dialog.progress_bar.value(), 0)
            self.assertFalse(dialog.cancel_button.isEnabled())
        finally:
            dialog.close()

    def test_setup_success_saves_only_validated_interpreter(self):
        from samstars_qgis.setup_dialog import SetupDialog
        from samstars_qgis.runtime import manager
        dialog = SetupDialog()
        try:
            with patch("samstars_qgis.setup_dialog.QgsSettings") as settings:
                dialog._succeeded({"python": "/tmp/validated/bin/python"})
                settings.return_value.setValue.assert_called_once_with(
                    manager.SETTING, "/tmp/validated/bin/python",
                )
            dialog._finished()
            self.assertEqual(dialog.progress_bar.value(), 100)
            self.assertIn("Engine ready", dialog.message.text())
        finally:
            dialog.close()

    def test_setup_uses_repair_for_managed_engine_and_respects_override(self):
        from samstars_qgis.setup_dialog import SetupDialog
        state = {
            "installed": True, "managed": True, "python": "/tmp/engine/bin/python",
            "path": "/tmp/engines", "platform": "Linux x86_64", "removable": True,
            "override": False, "error": "",
        }
        with patch("samstars_qgis.setup_dialog.manager.status", return_value=state):
            dialog = SetupDialog()
            try:
                self.assertEqual(dialog.install_button.text(), "Repair engine")
                self.assertTrue(dialog.remove_button.isEnabled())
                state["override"] = True
                dialog._refresh()
                self.assertFalse(dialog.install_button.isEnabled())
                self.assertFalse(dialog.existing_button.isEnabled())
                self.assertFalse(dialog.remove_button.isEnabled())
                self.assertIn("SAMSTARS_PYTHON", dialog.message.text())
            finally:
                dialog.close()

    def test_setup_thread_reports_progress_and_validates_before_activation(self):
        from qgis.PyQt.QtTest import QTest
        from samstars_qgis.setup_dialog import SetupDialog
        from samstars_qgis.runtime import manager
        dialog = SetupDialog()

        def operation(*, event, cancel):
            event({"type": "progress", "value": 45, "message": "Installing libraries"})
            event({"type": "log", "message": "Validation passed"})
            return {"python": "/tmp/validated/bin/python"}

        try:
            with patch("samstars_qgis.setup_dialog.QgsSettings") as settings:
                dialog._start(operation)
                deadline = time.monotonic() + 5
                while dialog._thread is not None and time.monotonic() < deadline:
                    QTest.qWait(10)
                self.assertIsNone(dialog._thread)
                settings.return_value.setValue.assert_called_once_with(
                    manager.SETTING, "/tmp/validated/bin/python",
                )
                self.assertEqual(dialog.progress_bar.value(), 100)
                self.assertIn("Validation passed", dialog.log.toPlainText())
        finally:
            if dialog._thread is not None:
                dialog._thread.cancel_event.set()
                dialog._thread.wait(2000)
                QTest.qWait(10)
            dialog.close()

    def test_setup_close_cancels_worker_without_changing_selection(self):
        from qgis.PyQt.QtTest import QTest
        from samstars_qgis.setup_dialog import SetupDialog
        from samstars_qgis.runtime.client import EngineCanceled
        dialog = SetupDialog()

        def operation(*, event, cancel):
            while not cancel():
                time.sleep(0.01)
            raise EngineCanceled("Setup canceled.")

        try:
            with patch("samstars_qgis.setup_dialog.QgsSettings") as settings:
                dialog._start(operation)
                dialog.reject()
                deadline = time.monotonic() + 5
                while dialog._thread is not None and time.monotonic() < deadline:
                    QTest.qWait(10)
                self.assertIsNone(dialog._thread)
                settings.assert_not_called()
                self.assertEqual(dialog.progress_bar.value(), 0)
                self.assertIn("canceled", dialog.message.text())
        finally:
            if dialog._thread is not None:
                dialog._thread.cancel_event.set()
                dialog._thread.wait(2000)
                QTest.qWait(10)
            dialog.close()

    def test_setup_removal_clears_only_a_removed_selection(self):
        from samstars_qgis.setup_dialog import SetupDialog
        from samstars_qgis.runtime import manager
        dialog = SetupDialog()
        try:
            with patch("samstars_qgis.setup_dialog.QgsSettings") as settings:
                settings.return_value.value.return_value = "/tmp/private-engine/bin/python"
                dialog._succeeded({"removed": ["/tmp/private-engine"]})
                settings.return_value.remove.assert_called_once_with(manager.SETTING)
            with patch("samstars_qgis.setup_dialog.QgsSettings") as settings:
                settings.return_value.value.return_value = "/tmp/external-env/bin/python"
                dialog._succeeded({"removed": ["/tmp/private-engine"]})
                settings.return_value.remove.assert_not_called()
        finally:
            dialog.close()

    @unittest.skipUnless(QGIS_AVAILABLE and Qgis.QGIS_VERSION_INT >= 40000, "Requires QGIS 4 and Qt6")
    def test_qt6_menu_lifecycle(self):
        from qgis.PyQt.QtWidgets import QMainWindow
        from samstars_qgis.samstars_plugin import SamStarsPlugin
        window = QMainWindow()
        iface = MagicMock()
        iface.mainWindow.return_value = window
        plugin = SamStarsPlugin(iface)
        try:
            plugin.initGui()
            plugin.initGui()
            iface.addPluginToMenu.assert_called_once()
            self.assertEqual(plugin.setup_action.text(), "Set up SAMStars…")
        finally:
            plugin.unload()
            window.close()
        iface.removePluginMenu.assert_called_once()


if __name__ == "__main__":
    unittest.main()
