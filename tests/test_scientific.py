import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

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

    def write_raster(self, path, count=1, crs="EPSG:32604", transform=None):
        with self.rasterio.open(path, "w", driver="GTiff", width=16, height=16, count=count,
                              dtype="float32", crs=crs, transform=transform or self.transform) as dataset:
            dataset.write(self.np.full((count, 16, 16), 10, dtype="float32"))

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
