"""Validation and log interpretation without scientific dependencies."""

import unittest

from support import ROOT
from samstars_qgis.runtime.training import training_options
from samstars_qgis.runtime.progress import TrainingProgress, clean_log


class TrainingOptionsTests(unittest.TestCase):
    def test_window_units_and_automatic_or_manual_stride(self):
        self.assertEqual(training_options({}), {"chip_size": 512, "chip_stride": 384})
        self.assertEqual(training_options({"chip_size": 256, "chip_stride": 100})["chip_stride"], 100)
        metres = {"sam_window_size_m": 25.6}
        self.assertEqual(training_options(metres), metres)
        self.assertEqual(training_options({**metres, "chip_stride": 192})["chip_stride"], 192)
        with self.assertRaisesRegex(ValueError, "not both"):
            training_options({**metres, "chip_size": 256})
        with self.assertRaisesRegex(ValueError, "leave gaps"):
            training_options({"chip_size": 256, "chip_stride": 257})

    def test_inactive_values_are_ignored_without_mutating_the_request(self):
        values = {"thin_seeds": False, "thin_dist": -100, "stardist_window_overlap_m": -100}
        options = training_options(values)
        self.assertNotIn("thin_dist", options)
        self.assertNotIn("stardist_window_overlap_m", options)
        self.assertEqual(values["thin_dist"], -100)

    def test_alpha_options_cannot_be_silently_lost_on_stable(self):
        training_options({}, "0.2.0")
        self.assertNotIn("fg_extra", training_options({"fg_extra": 80}, "0.2.0"))
        self.assertEqual(training_options({"fg_extra": 80}, "0.3.0a1")["fg_extra"], 80)
        for values in ({"fg_extra": 0}, {"fg_extra": 120}, {"stardist_patch_size": 256}, {"sam_window_size_m": 50}, {"chip_size": 256}):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "require SAMStars 0.3.0a1"):
                    training_options(values, "0.2.0")
                training_options(values, "0.3.0a1")

    def test_numeric_ranges_and_types(self):
        invalid = (
            ("sam_lr", 0), ("sam_lr", float("nan")), ("sam_lr", float("inf")),
            ("sam_epochs", 0), ("sam_epochs", 1.5), ("sam_min_iters", -1),
            ("stardist_min_iters", True), ("fg_extra", -1), ("fg_extra", "80"),
            ("seed_min_dist_px", 0), ("seed_gauss_sigma", -1), ("seed_min_samples", 0),
            ("seed_merge_radius", 0), ("seed_d_min", -1), ("seed_min_canopy_height", -1),
            ("thin_dist", -1), ("crop_smooth_sigma", -1), ("smooth_sigma", -1), ("post_smooth_sigma", -1),
            ("val_fraction", 1), ("val_fraction", -.1), ("stardist_patch_size", 255),
            ("stardist_window_size_m", -1), ("stardist_dedup_iou_thr", 1),
            ("stardist_input_key", "unknown"), ("thin_seeds", 1), ("fast", "False"),
        )
        for name, value in invalid:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                training_options({name: value})
        training_options({"sam_min_iters": 0, "fg_extra": 0, "val_fraction": 0, "seed_gauss_sigma": 0, "post_smooth_sigma": 0, "stardist_patch_size": 16})

    def test_refinement_overlap_requires_smaller_window(self):
        for overlap in (-1, 50, 51):
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                training_options({"stardist_window_size_m": 50, "stardist_window_overlap_m": overlap})
        options = training_options({"stardist_window_size_m": 50, "stardist_window_overlap_m": 5})
        self.assertEqual(options["stardist_window_overlap_m"], 5)


class TrainingProgressTests(unittest.TestCase):
    def test_sam_epoch_completion_and_stage_eta(self):
        progress = TrainingProgress()
        self.assertIn("started", progress.consume("chips: epochs=15 (iters/epoch=20)", now=100))
        status = progress.consume("chips epoch 3/15 loss=0.1234", now=130)
        self.assertIn("3/15 epochs complete (20%)", status)
        self.assertIn("2m remaining in this stage", status)
        self.assertIn("StarDist training follow", progress.consume("chips epoch 15/15 loss=0.1", now=250))

    def test_stardist_distinguishes_batch_count_from_epoch_count(self):
        progress = TrainingProgress()
        self.assertIsNone(progress.consume("32/100 ==== loss: 2.27", now=1))
        self.assertNotIn("remaining", progress.consume("Epoch 27/50", now=100))
        self.assertIn("12m remaining", progress.consume("Epoch 28/50", now=130))
        line = "\x1b[1m32/100\x1b[0m \x1b[32m=====\x1b[0m 19s 283ms/step - loss: 2.2708"
        status = progress.consume(line, now=140)
        self.assertIn("epoch 28/50, step 32/100 (54.6%", status)
        self.assertNotIn("\x1b", clean_log(line))
        progress.consume("Epoch 50/50", now=800)
        self.assertIn("validation, threshold optimization", progress.consume("100/100 ==== loss: 1.2", now=830))
        self.assertIn("bundle is saved", progress.consume("SAMStars finished.", now=850))

    def test_missing_timing_and_malformed_logs_do_not_invent_eta(self):
        progress = TrainingProgress()
        self.assertNotIn("remaining", progress.consume("chips epoch 2/10 loss=0.3", now=10))
        for line in ("Epoch 0/0", "Epoch 51/50", "0/0 loss: 0", "Reading raster 32/100", "unrelated"):
            self.assertIsNone(progress.consume(line, now=10))


if __name__ == "__main__":
    unittest.main()
