import atexit
import gc
from pathlib import Path
import sys
import tempfile
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

    def test_provider_registers_exactly_three_tools_without_ml_imports(self):
        from samstars_qgis.provider import SamStarsProvider
        before = set(sys.modules)
        provider = SamStarsProvider()
        registry = QgsApplication.processingRegistry()
        self.assertTrue(registry.addProvider(provider))
        try:
            self.assertEqual({algorithm.id() for algorithm in provider.algorithms()}, {
                "samstars:segment", "samstars:train", "samstars:checkdependencies",
            })
            for algorithm in provider.algorithms():
                self.assertTrue(Path(algorithm.helpUrl().split("#")[0].removeprefix("file://")).is_file())
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
            self.assertEqual(dialog.windowTitle(), "SAMStars setup")
            self.assertFalse(dialog.cancel.isEnabled())
            self.assertIsNone(dialog._thread)
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
