import importlib.util
import importlib.metadata
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from support import ROOT
from samstars_qgis.runtime import manager, worker

SCIENTIFIC_AVAILABLE = importlib.util.find_spec("rasterio") is not None and importlib.util.find_spec("samstars") is not None


@unittest.skipUnless(SCIENTIFIC_AVAILABLE, "Requires the separate SAMStars scientific environment")
class ScientificTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        self.np = np
        self.rasterio = rasterio
        self.transform = from_origin(500000, 2200016, 1, 1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = {}
        for key, count in (("img", 4), ("chm", 1), ("den", 1)):
            path = self.root / f"{key}.tif"
            self.write_raster(path, count=count)
            self.payload[key] = str(path)

    def write_raster(self, path, count=1, crs="EPSG:32604", transform=None, size=16):
        with self.rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=count,
                              dtype="float32", crs=crs, transform=transform or self.transform) as dataset:
            dataset.write(self.np.full((count, size, size), 10, dtype="float32"))

    def test_custom_sam_window_reaches_real_alpha_training_chips(self):
        if importlib.metadata.version("samstars") != "0.3.0a1":
            self.skipTest("Custom SAM windows require the experimental engine")
        import geopandas as gpd
        from rasterio.transform import from_origin
        from shapely.geometry import box
        from samstars import api
        from samstars.training import sam_training

        tiles = self.root / "tiles"
        tile = tiles / "scene"
        tile.mkdir(parents=True)
        payload = {"out": str(self.root / "model"), "storage": "temp", "device": "cpu", "chip_size": 256}
        for key, count in (("img", 3), ("chm", 1), ("den", 1)):
            path = tile / f"{key}.tif"
            self.write_raster(path, count=count, size=320, transform=from_origin(500000, 2200032, .1, .1))
            payload[key] = str(path)
        crowns = tile / "crowns.gpkg"
        gpd.GeoDataFrame({"tree_id": [1]}, geometry=[box(500004, 2200004, 500028, 2200028)],
                         crs="EPSG:32604").to_file(crowns, driver="GPKG", engine="pyogrio")
        payload["crowns"] = str(crowns)
        payload["sam_base_ckpt"] = str(self.root / "unused-checkpoint.pth")

        class BeforeOptimization(Exception):
            pass

        expected_stride = 192

        def inspect_chips(**kwargs):
            self.assertEqual(kwargs["img_size"], 256)
            self.assertEqual(kwargs["preprocessing"]["sam_window_size_px"], 256)
            self.assertEqual(kwargs["preprocessing"]["sam_stride_px"], expected_stride)
            self.assertAlmostEqual(kwargs["preprocessing"]["sam_window_size_m"], 25.6)
            self.assertEqual(kwargs["epochs"], 2)
            self.assertEqual(kwargs["min_iters"], 3)
            self.assertEqual(kwargs["lr"], .0001)
            self.assertEqual(kwargs["fg_extra"], 0)
            self.assertEqual(kwargs["preprocessing"]["stardist_patch_size_px"], 128)
            images = list((kwargs["chip_dir"] / "images").glob("*.npy"))
            self.assertGreater(len(images), 0)
            for image in images:
                self.assertEqual(self.np.load(image).shape, (3, 256, 256))
                self.assertEqual(self.np.load(kwargs["chip_dir"] / "masks" / image.name).shape, (256, 256))
            raise BeforeOptimization()

        # Isolate window handling: reuse aligned fixtures, build real chips, and
        # stop before loading a checkpoint or running neural optimization.
        payload.update(sam_epochs=2, sam_min_iters=3, sam_lr=.0001, fg_extra=0, stardist_patch_size=128)
        for units, options, expected_stride in (
            ("pixels", {"chip_size": 256}, 192),
            ("metres", {"sam_window_size_m": 25.6, "chip_stride": 128}, 128),
        ):
            with self.subTest(units=units), \
                 patch.object(api, "_resolve_training_tiles", return_value=tiles), \
                 patch.object(sam_training, "train_decoder", side_effect=inspect_chips) as decoder:
                request = {**payload, **options, "out": str(self.root / units)}
                if units == "metres":
                    request.pop("chip_size")
                with self.assertRaises(BeforeOptimization):
                    worker.run_job("train", request)
                decoder.assert_called_once()

    def test_real_rasters_accept_model_band_defaults(self):
        worker.validate_rasters(self.payload, {"rgb_bands": [3, 2, 1], "nir_band": 4})

    def test_stored_model_band_is_checked_against_actual_image(self):
        with self.assertRaisesRegex(ValueError, "available 4 bands"):
            worker.validate_rasters(self.payload, {"rgb_bands": [1, 2, 5]})

    def test_geographic_image_is_rejected(self):
        self.write_raster(self.payload["img"], count=4, crs="EPSG:4326")
        with self.assertRaisesRegex(ValueError, "projected CRS"):
            worker.validate_rasters(self.payload)

    def test_nonsquare_image_is_rejected(self):
        from rasterio.transform import from_origin
        self.write_raster(self.payload["img"], count=4, transform=from_origin(500000, 2200016, 1, 2))
        with self.assertRaisesRegex(ValueError, "square pixels"):
            worker.validate_rasters(self.payload)

    def test_nonoverlapping_chm_is_rejected(self):
        from rasterio.transform import from_origin
        self.write_raster(self.payload["chm"], transform=from_origin(600000, 2200016, 1, 1))
        with self.assertRaisesRegex(ValueError, "does not overlap"):
            worker.validate_rasters(self.payload)

    def test_multiband_density_is_rejected(self):
        self.write_raster(self.payload["den"], count=2)
        with self.assertRaisesRegex(ValueError, "single-band"):
            worker.validate_rasters(self.payload)

    def test_chm_scale_must_be_applied(self):
        with self.rasterio.open(self.payload["chm"], "r+") as raster:
            raster.scales = [0.1]
        with self.assertRaisesRegex(ValueError, "scale/offset"):
            worker.validate_rasters(self.payload)

    def test_real_engine_dependency_check(self):
        result = manager.check_engine(sys.executable)
        self.assertTrue(result["ready"])
        self.assertEqual(result["python"], str(Path(sys.executable).absolute()))
        self.assertTrue(any(item["name"] == "CPU operations" for item in result["checks"]))
        self.assertTrue(any(item["name"] == "Geospatial operations" for item in result["checks"]))


if __name__ == "__main__":
    unittest.main()
