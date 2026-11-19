"""Workflow contracts; scientific cases run only in the separate engine."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from support import ROOT
from samstars_qgis.runtime import client, steps, worker

SCIENTIFIC_AVAILABLE = importlib.util.find_spec("rasterio") is not None and importlib.util.find_spec("samstars") is not None


class StepContractTests(unittest.TestCase):
    def test_alignment_protects_optional_inputs_and_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            reference = root / "reference.tif"
            reference.touch()
            intensity = inputs / "intensity.tif"
            intensity.write_bytes(b"keep me")
            with patch.object(steps, "align_rasters") as align, self.assertRaisesRegex(ValueError, "separate"):
                steps.run_step("alignrasters", {"reference": str(reference), "intensity": str(intensity), "out": str(inputs)},
                               emit=lambda *args: None, validate_output=worker.validate_output, claim_output=worker.claim_output)
            align.assert_not_called()
            self.assertEqual(intensity.read_bytes(), b"keep me")

    def test_refinement_defaults_and_explicit_whole_raster(self):
        defaults = {"stardist_prob_thr": .4, "stardist_nms_thr": .3,
                    "stardist_window_size_m": 50, "stardist_window_overlap_m": 5}
        self.assertEqual(steps.refinement_settings({}, defaults)["stardist_prob_thr"], .4)
        result = steps.refinement_settings({"stardist_window_size_m": 0, "stardist_window_overlap_m": 0}, defaults)
        self.assertEqual(result["stardist_window_size_m"], 0)
        for overrides in ({"stardist_prob_thr": .5}, {"stardist_window_size_m": 5},
                          {"stardist_dedup_iou_thr": 1}, {"min_area": float("nan")}, {"edge_buffer": -1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                steps.refinement_settings(overrides, defaults)

    def test_records_resolve_relative_paths_and_reject_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raster = root / "input.tif"
            raster.touch()
            path = root / "proposal_records.json"
            path.write_text(json.dumps([{key: "input.tif" for key in steps.RASTER_FIELDS}]))
            result = steps.read_proposal_records(path)
            self.assertEqual(result[0]["index"], str(raster))
            raster.unlink()
            with self.assertRaisesRegex(ValueError, "missing|Missing"):
                steps.read_proposal_records(path)

    def test_refine_protects_rasters_referenced_by_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            raster = inputs / "saved.tif"
            raster.write_bytes(b"keep me")
            path = root / "records.json"
            path.write_text(json.dumps([{key: str(raster) for key in steps.RASTER_FIELDS}]))
            with patch.object(steps, "refine") as refine, self.assertRaisesRegex(ValueError, "separate"):
                steps.run_step("refine", {"records": str(path), "model": str(root / "model"), "out": str(inputs)},
                               emit=lambda *args: None, validate_output=worker.validate_output, claim_output=worker.claim_output)
            refine.assert_not_called()
            self.assertEqual(raster.read_bytes(), b"keep me")

    def test_failed_step_releases_output_claim_without_removing_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root
            image = root / "img.tif"
            image.write_bytes(b"original input")
            record = root / "workflow.json"
            record.write_bytes(b"alignment settings")

            def fail(p, destination, emit):
                (destination / "cost.tif").write_bytes(b"partial result")
                raise RuntimeError("interrupted")

            with patch.object(steps, "cost_surface", side_effect=fail), self.assertRaisesRegex(RuntimeError, "interrupted"):
                steps.run_step("costsurface", {"img": str(image), "chm": str(root / "chm"), "out": str(out)},
                               emit=lambda *args: None, validate_output=worker.validate_output, claim_output=worker.claim_output)
            self.assertEqual(image.read_bytes(), b"original input")
            self.assertEqual(record.read_bytes(), b"alignment settings")
            self.assertEqual((out / "cost.tif").read_bytes(), b"partial result")
            self.assertFalse((out / ".samstars-running").exists())
            self.assertFalse((out / "cost.workflow.json").exists())

    def test_cost_and_seed_output_conflicts_fail_before_processing(self):
        cases = (
            ("costsurface", "cost_surface", ("cost.tif", "cost.workflow.json")),
            ("generateseeds", "generate_seeds", ("seeds.gpkg", "seeds.workflow.json")),
        )
        for operation, function, filenames in cases:
            for name in filenames:
                with self.subTest(step=operation, filename=name), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    target = root / name
                    target.write_bytes(b"existing result or settings")
                    payload = {key: str(root / f"{key}.tif") for key in ("img", "chm", "den", "cost")}
                    payload["out"] = str(root)
                    with patch.object(steps, function) as process, self.assertRaisesRegex(ValueError, name):
                        steps.run_step(operation, payload, emit=lambda *args: None,
                                       validate_output=worker.validate_output, claim_output=worker.claim_output)
                    process.assert_not_called()
                    self.assertEqual(target.read_bytes(), b"existing result or settings")


@unittest.skipUnless(SCIENTIFIC_AVAILABLE, "Requires the separate SAMStars engine")
class ScientificStepTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        self.np, self.rio = np, rasterio
        self.temp = tempfile.TemporaryDirectory(prefix="samstars-step-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.transform = from_origin(500000, 2200032, 1, 1)
        y, x = np.mgrid[:32, :32]
        self.chm = (18 * np.exp(-((x - 8) ** 2 + (y - 8) ** 2) / 18)
                    + 9 * np.exp(-((x - 24) ** 2 + (y - 24) ** 2) / 18)).astype("float32")
        self.payload = {
            "img": self.write("image.tif", np.stack((x + 1, y + 2, x + y + 3, 50 - x))),
            "chm": self.write("chm.tif", self.chm),
            "den": self.write("density.tif", np.zeros((32, 32))),
            "cost": self.write("cost.tif", np.zeros((32, 32))),
        }

    def write(self, name, data, *, transform=None, crs="EPSG:32604"):
        data = self.np.asarray(data, dtype="float32")
        if data.ndim == 2:
            data = data[None]
        path = self.root / name
        with self.rio.open(path, "w", driver="GTiff", count=data.shape[0], height=data.shape[1], width=data.shape[2],
                           dtype="float32", crs=crs, transform=transform or self.transform, nodata=-9999) as raster:
            raster.write(data)
        return str(path)

    def run_step(self, operation, name, **extra):
        return steps.run_step(operation, {**self.payload, "out": str(self.root / name), **extra},
                              emit=lambda *args: None, validate_output=worker.validate_output, claim_output=worker.claim_output)

    def points(self):
        import geopandas as gpd
        from shapely.geometry import Point
        path = self.root / "edited.gpkg"
        gpd.GeoDataFrame({"note": ["moved", "outside"]}, geometry=[Point(500009.5, 2200022.5), Point(600000, 2200020)],
                         crs="EPSG:32604").to_file(path, driver="GPKG", engine="pyogrio", index=False)
        return str(path)

    def test_isolated_alignment_handles_resolution_crs_and_nodata(self):
        from rasterio.transform import from_bounds, from_origin
        from rasterio.warp import transform_bounds
        imagery = self.write("coarse_image.tif", self.np.stack([self.np.full((16, 16), i) for i in range(1, 9)]),
                             transform=from_origin(500000, 2200032, 2, 2))
        with self.rio.open(imagery, "r+") as raster:
            raster.set_band_description(8, "NIR2")
        intensity = self.write("partial_intensity.tif", self.np.full((8, 8), 12),
                               transform=from_origin(500000, 2200032, 2, 2))
        with self.rio.open(self.payload["chm"], "r+") as raster:
            chm = raster.read(1)
            chm[0, 0] = -9999
            raster.write(chm, 1)
        original_chm = Path(self.payload["chm"]).read_bytes()
        bounds = transform_bounds("EPSG:32604", "EPSG:4326", 500000, 2200000, 500032, 2200032)
        den = self.write("geographic.tif", self.np.full((16, 16), 8), crs="EPSG:4326", transform=from_bounds(*bounds, 16, 16))
        result = client.call_with_python(sys.executable, "alignrasters", {
            "reference": self.payload["chm"], "img": imagery, "chm": self.payload["chm"],
            "den": den, "intensity": intensity, "out": str(self.root / "aligned"),
        })
        self.assertEqual(result["REFERENCE_RASTER"], self.payload["chm"])
        self.assertEqual(Path(self.payload["chm"]).read_bytes(), original_chm)
        for key in ("ALIGNED_IMAGE", "ALIGNED_CHM", "ALIGNED_DENSITY", "ALIGNED_INTENSITY"):
            with self.rio.open(result[key]) as raster:
                self.assertEqual(raster.transform, self.transform)
                self.assertEqual(raster.shape, (32, 32))
                self.assertEqual(raster.crs.to_epsg(), 32604)
                if key == "ALIGNED_CHM":
                    self.np.testing.assert_array_equal(raster.read(1), chm)
                elif key == "ALIGNED_IMAGE":
                    self.assertEqual(raster.count, 8)
                    self.assertEqual(raster.descriptions[7], "NIR2")
                    for band in range(1, 9):
                        self.np.testing.assert_allclose(raster.read(band), band)
                elif key == "ALIGNED_INTENSITY":
                    self.assertEqual(raster.read(1)[4, 4], 12)
                    self.assertTrue(raster.read(1, masked=True).mask[24, 24])
                else:
                    self.assertAlmostEqual(float(raster.read(1)[16, 16]), 8)
        self.assertTrue((self.root / "aligned/workflow.json").is_file())

    def test_alignment_uses_multiband_reference_with_only_density_selected(self):
        from rasterio.transform import from_origin
        reference = self.write("reference.tif", self.np.zeros((4, 16, 16)), transform=from_origin(500000, 2200032, 2, 2))
        result = steps.run_step("alignrasters", {"reference": reference, "den": self.payload["den"], "out": str(self.root / "aligned")},
                                emit=lambda *args: None, validate_output=worker.validate_output, claim_output=worker.claim_output)
        self.assertEqual(set(result), {"OUTPUT", "REFERENCE_RASTER", "ALIGNED_DENSITY"})
        with self.rio.open(result["ALIGNED_DENSITY"]) as raster:
            self.assertEqual(raster.shape, (16, 16))
            self.assertEqual(raster.transform.a, 2)
            self.assertEqual(raster.count, 1)
        self.assertEqual({path.name for path in (self.root / "aligned").iterdir()}, {"den.tif", "workflow.json"})

    def test_alignment_rejects_empty_selection_bad_reference_and_nonoverlap(self):
        from rasterio.transform import from_origin
        def run(name, **params):
            return steps.run_step("alignrasters", {"reference": self.payload["chm"], "out": str(self.root / name), **params},
                                  emit=lambda *args: None, validate_output=worker.validate_output, claim_output=worker.claim_output)
        with self.assertRaisesRegex(ValueError, "at least one"):
            run("no-input")
        reference = self.write("geographic_reference.tif", self.chm, crs="EPSG:4326")
        with self.assertRaisesRegex(ValueError, "projected CRS"):
            run("bad-reference", reference=reference, den=self.payload["den"])
        outside = self.write("outside.tif", self.chm, transform=from_origin(600000, 2200032, 1, 1))
        with self.assertRaisesRegex(ValueError, "does not overlap the reference"):
            run("no-overlap", intensity=outside)

    def test_cost_surface_weights_and_grid_validation(self):
        first = self.run_step("costsurface", "height-cost", weights=[1, 0, 0])
        second = self.run_step("costsurface", "ndvi-cost", weights=[0, 1, 0], nir_band=4)
        with self.rio.open(first["COST"]) as a, self.rio.open(second["COST"]) as b:
            self.assertEqual(a.count, 1)
            self.assertGreaterEqual(float(a.read(1).min()), 0)
            self.assertLessEqual(float(a.read(1).max()), 1)
            self.assertFalse(self.np.allclose(a.read(1), b.read(1)))
        with self.assertRaisesRegex(ValueError, "NIR"):
            self.run_step("costsurface", "bad-weights", weights=[0, 1, 0])
        with self.assertRaisesRegex(ValueError, "band indices"):
            self.run_step("costsurface", "bad-band", rgb_bands=[1, 2, 9])
        shifted = self.write("shifted.tif", self.chm, transform=self.transform * self.rio.Affine.translation(1, 0))
        with self.assertRaisesRegex(ValueError, "pixel grid"):
            self.run_step("costsurface", "bad-grid", chm=shifted)

    def test_seed_parameters_change_real_saved_points(self):
        import geopandas as gpd
        settings = {"gauss_sigma": 0, "min_dist_px": 2, "merge_radius": 1, "min_canopy_height": 2}
        low = self.run_step("generateseeds", "low", seed_settings={**settings, "d_min": 5})
        high = self.run_step("generateseeds", "high", seed_settings={**settings, "d_min": 12})
        low_points, high_points = (gpd.read_file(result["SEEDS"]) for result in (low, high))
        self.assertEqual(len(low_points), 2)
        self.assertEqual(len(high_points), 1)
        self.assertEqual(low_points.crs.to_epsg(), 32604)
        with self.assertRaisesRegex(ValueError, "No seed points"):
            self.run_step("generateseeds", "empty", seed_settings={**settings, "d_min": 100})

    def test_isolated_cost_and_seeds_share_input_folder_without_overwriting(self):
        import geopandas as gpd

        # Keep an earlier result and alignment record beside the source rasters.
        previous_cost = self.root / "previous_cost.tif"
        Path(self.payload["cost"]).rename(previous_cost)
        (self.root / "workflow.json").write_text('{"step": "alignrasters"}')
        originals = {path: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        cost_payload = {key: self.payload[key] for key in ("img", "chm")}
        cost_payload.update(out=str(self.root), weights=[1, 0, 0])
        cost = client.call_with_python(sys.executable, "costsurface", cost_payload)
        with self.rio.open(cost["COST"]) as raster:
            self.assertEqual(raster.transform, self.transform)
            self.assertGreaterEqual(float(raster.read(1).min()), 0)
            self.assertLessEqual(float(raster.read(1).max()), 1)
        seed_payload = {
            "chm": self.payload["chm"], "den": self.payload["den"], "cost": cost["COST"], "out": str(self.root),
            "seed_settings": {"d_min": 5, "gauss_sigma": 0, "min_dist_px": 2, "merge_radius": 1, "min_canopy_height": 2},
        }
        originals[Path(cost["COST"])] = Path(cost["COST"]).read_bytes()
        seeds = client.call_with_python(sys.executable, "generateseeds", seed_payload)
        points = gpd.read_file(seeds["SEEDS"])
        self.assertEqual(len(points), 2)
        self.assertEqual(points.crs.to_epsg(), 32604)
        for name, operation, payload, result in (
            ("cost", "costsurface", cost_payload, cost), ("seeds", "generateseeds", seed_payload, seeds),
        ):
            record = self.root / f"{name}.workflow.json"
            document = json.loads(record.read_text())
            self.assertEqual(document["step"], operation)
            self.assertEqual(document["parameters"], payload)
            self.assertEqual(document["outputs"], result)
        originals.update({path: path.read_bytes() for path in (
            Path(seeds["SEEDS"]), self.root / "cost.workflow.json", self.root / "seeds.workflow.json",
        )})
        for operation, payload, filename in (
            ("costsurface", cost_payload, "cost.tif"), ("generateseeds", seed_payload, "seeds.gpkg"),
        ):
            with self.subTest(step=operation), self.assertRaisesRegex(client.EngineError, filename):
                client.call_with_python(sys.executable, operation, payload)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        self.assertFalse((self.root / ".samstars-running").exists())

    def test_selected_points_are_clipped_and_missing_data_rejected(self):
        crs, bounds = steps.aligned({key: self.payload[key] for key in ("chm", "cost")})
        events = []
        path = self.points()
        result = steps._prepare_points(path, crs, bounds, self.payload["chm"], self.payload["cost"], events.append)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.geometry.x.iloc[0], 500009.5)
        self.assertIn("Ignored 1", events[0])
        with self.rio.open(self.payload["cost"], "r+") as raster:
            values = raster.read(1)
            values[9, 9] = -9999
            raster.write(values, 1)
        with self.assertRaisesRegex(ValueError, "missing CHM or cost"):
            steps._prepare_points(path, crs, bounds, self.payload["chm"], self.payload["cost"], events.append)

    def test_proposals_use_prepared_rgb_and_user_seeds_without_regeneration(self):
        import geopandas as gpd
        model = types.SimpleNamespace(defaults={"rgb_bands": [3, 2, 1], "smooth_sigma": 1}, decoder_path=self.root / "decoder.pth")

        def infer(records_path, out, checkpoint, **kwargs):
            record = json.loads(records_path.read_text())[0]
            tile = Path(record["img_path"]).parent
            with self.rio.open(tile / "img.tif") as prepared, self.rio.open(self.payload["img"]) as original:
                self.assertEqual(prepared.count, 3)
                self.np.testing.assert_array_equal(prepared.read(), original.read([3, 2, 1]))
            points = gpd.read_file(tile / "seeds.gpkg")
            self.assertEqual(len(points), 1)
            self.assertEqual(points.geometry.x.iloc[0], 500009.5)
            self.assertEqual(kwargs["smooth_sigma"], 1)
            out.mkdir()
            # Substitute inference only. Real preparation, records, and rasters
            # still cross the same boundary used by SAM.
            records = [{key: self.payload["chm" if key == "chm" else "cost"] for key in steps.RASTER_FIELDS}]
            (out / "proposal_records.json").write_text(json.dumps(records))

        with patch.object(steps, "_model", return_value=model), \
             patch("samstars.segmentation.proposals.run_tile_proposals", side_effect=infer), \
             patch("samstars.features.canonical_seeds", side_effect=AssertionError("seeds regenerated")), \
             patch("samstars.features.build_cost_surface", side_effect=AssertionError("cost regenerated")):
            result = self.run_step("proposals", "proposals-out", seeds=self.points(), model=str(self.root / "model"))
        self.assertTrue(Path(result["PROPOSAL_RECORDS"]).is_file())

    def test_proposal_batching_defaults_and_saved_window_settings(self):
        settings = {"version": 1, "resolution_m": 1, "sam_window_size_px": 32,
                    "sam_window_size_m": 32, "sam_stride_px": 24, "sam_encoder_size": 1024,
                    "padding": "constant_zero", "rgb_scaling": "valid_percentile_2_98",
                    "stardist_patch_size_px": 32}
        for current in (False, True):
            model = types.SimpleNamespace(defaults={}, decoder_path=self.root / "decoder.pth")
            if current:
                model.preprocessing = settings
            for fast in (None, False, True):
                with self.subTest(current=current, fast=fast):
                    expected = True if fast is None else fast

                    def infer(records_path, out, checkpoint, **kwargs):
                        self.assertIs(kwargs["fast"], expected)
                        if current:
                            self.assertEqual(kwargs["preprocessing"], settings)
                        else:
                            self.assertNotIn("preprocessing", kwargs)
                        out.mkdir()
                        records = [{key: self.payload["chm" if key == "chm" else "cost"] for key in steps.RASTER_FIELDS}]
                        (out / "proposal_records.json").write_text(json.dumps(records))

                    with patch.object(steps, "_model", return_value=model), \
                         patch("samstars.segmentation.proposals.run_tile_proposals", side_effect=infer):
                        self.run_step("proposals", f"proposals-{current}-{fast}",
                                      seeds=self.points(), model=str(self.root / "model"),
                                      **({} if fast is None else {"fast": fast}))

    def test_real_stardist_refinement_uses_saved_proposals_without_sam(self):
        import geopandas as gpd
        from samstars.api import _BUNDLE_FORMAT
        from stardist.models import Config2D, StarDist2D
        # A tiny untrained network verifies serialization, CLI invocation,
        # channel handling, and GeoPackage output; it cannot measure accuracy.
        config = Config2D(n_rays=8, n_channel_in=2, grid=(2, 2), unet_n_depth=1,
                          unet_n_filter_base=4, train_patch_size=(32, 32))
        model = StarDist2D(config, name="model", basedir=str(self.root))
        model.keras_model.set_weights([self.np.zeros_like(value) for value in model.keras_model.get_weights()])
        for name, bias in (("prob", 2), ("dist", 3)):
            layer = model.keras_model.get_layer(name)
            weights = layer.get_weights()
            weights[-1][:] = bias
            layer.set_weights(weights)
        model.keras_model.save_weights(str(self.root / "model/weights_best.h5"))
        (self.root / "model/decoder.pth").touch()  # Must never be opened by refinement.
        manifest = {"bundle_format": _BUNDLE_FORMAT, "bundle_version": 1, "decoder": "decoder.pth",
                    "config": "config.json", "weights": "weights_best.h5",
                    "defaults": {"stardist_use_aux": True, "stardist_input_key": "index"}}
        (self.root / "model/manifest.json").write_text(json.dumps(manifest))
        records = self.root / "proposal_records.json"
        records.write_text(json.dumps([{key: self.payload["chm" if key == "chm" else "cost"] for key in steps.RASTER_FIELDS}]))
        with patch("samstars.segmentation.proposals.run_tile_proposals", side_effect=AssertionError("SAM reran")):
            result = self.run_step("refine", "refined", model=str(self.root / "model"), records=str(records),
                                   stardist_prob_thr=.5, stardist_nms_thr=.3,
                                   stardist_window_size_m=16, stardist_window_overlap_m=4, stardist_dedup_iou_thr=.5)
        crowns = gpd.read_file(result["CROWNS"])
        self.assertGreater(len(crowns), 0)
        self.assertTrue(crowns.geometry.is_valid.all())
        self.assertEqual(crowns.crs.to_epsg(), 32604)
        self.assertTrue((self.root / "refined/proposal_scale_stats.json").is_file())

    def test_geometry_repair_retains_crown_identity_and_area(self):
        import geopandas as gpd
        from shapely.geometry import Polygon
        # Two raster cells touching at a corner form a self-touching polygon.
        geometry = Polygon([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (1, 2), (1, 1), (0, 1), (0, 0)])
        self.assertFalse(geometry.is_valid)
        path = self.root / "crowns.gpkg"
        gpd.GeoDataFrame({"label": [7], "score": [.8], "source": ["scene"]}, geometry=[geometry],
                         crs="EPSG:32604").to_file(path, layer="crowns", driver="GPKG", index=False)
        events = []
        steps._finish_crowns(path, lambda message, *args: events.append(message))
        crowns = gpd.read_file(path)
        self.assertEqual(len(crowns), 1)
        self.assertEqual(crowns.label.iloc[0], 7)
        self.assertEqual(crowns.score.iloc[0], .8)
        self.assertTrue(crowns.geometry.is_valid.all())
        self.assertAlmostEqual(crowns.geometry.area.iloc[0], geometry.area)
        self.assertTrue(any("Repaired 1" in event for event in events))


if __name__ == "__main__":
    unittest.main()
