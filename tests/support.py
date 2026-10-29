"""Load the checkout under its installable plugin name, without changing sys.path."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if "samstars_qgis" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "samstars_qgis", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
