"""Opt-in hardware check, run with the engine's Python (no model download)."""

import json
import os
import sys
import unittest

from support import ROOT
from samstars_qgis.runtime.client import call_with_python


@unittest.skipUnless(os.environ.get("SAMSTARS_TEST_GPU"), "Set SAMSTARS_TEST_GPU=cuda or mps to test real hardware")
class GpuHardwareTests(unittest.TestCase):
    def test_gpu_forward_and_backward_operations(self):
        device = os.environ["SAMSTARS_TEST_GPU"]
        self.assertIn(device, ("cuda", "mps"))
        result = call_with_python(sys.executable, "check_gpu", {"device": device}, timeout=180)
        self.assertTrue(result["ready"], json.dumps(result, indent=2))
        self.assertEqual(result["device"], device)
        self.assertTrue(result["hardware"])


if __name__ == "__main__":
    unittest.main()
