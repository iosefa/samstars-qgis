# Verification

Run the standard-library tests with Python 3.10–3.12:

```bash
python3 -m unittest discover -s tests -p 'test_runtime.py' -v
```

Run the QGIS tests using the Python shipped with QGIS:

```bash
QT_QPA_PLATFORM=offscreen /usr/bin/python3 -m unittest discover -s tests -p 'test_qgis.py' -v
```

These test actual Processing registration, input validation, selection-aware
polygon export, output loading, setup construction, and the minimum-version
guard. Neural model calls are substituted at the engine boundary. The shared
Processing APIs can be checked with QGIS 3.44, but plugin startup requires 4.0.
The Qt6 menu lifecycle test runs only on QGIS 4+.

For optional real scientific dependency and raster validation checks, use a
separate supported Python environment with `samstars==0.2.0`:

```bash
/path/to/engine/bin/python -m unittest discover -s tests -p 'test_scientific.py' -v
```

No tests download checkpoints or run model training. Full model runs require
the user's imagery, CHM, density, annotations, and a ViT-L checkpoint or trained
bundle. GPU, Windows, and macOS behavior require platform testing. Dependency
resolution is not a locked, reproducible scientific environment.

Build the release artifact with `python3 scripts/package_plugin.py`; packaging
is tested for a valid `samstars_qgis/` root, required files, and QGIS 4.0 metadata.

Initial verification used Linux x64, QGIS 3.44.7 / Qt 5.15.13 / Python 3.12.3,
plus a disposable Python 3.10 environment with SAMStars 0.2.0, PyTorch 2.10.0,
TensorFlow 2.18.1, StarDist 0.9.1, and Rasterio 1.4.3. Real CPU and geospatial
operations passed. The existing conda geospatial packages had a CRS lookup
problem, so self-contained Rasterio, PyProj, and Pyogrio wheels were installed
only in the disposable environment. QGIS 4/Qt6 and complete neural model runs
have not been executed here; do not treat the QGIS 3.44 checks as QGIS 4 runtime
certification. The full private installer has not been exercised end to end.
