# Verification

Run the standard-library tests with Python 3.10–3.12:

```bash
python3 -m unittest discover -s tests -p 'test_runtime.py' -v
python3 -m unittest discover -s tests -p 'test_steps.py' -v
```

Run the QGIS tests using the Python shipped with QGIS:

```bash
QT_QPA_PLATFORM=offscreen /usr/bin/python3 -m unittest discover -s tests -p 'test_qgis.py' -v
```

These test actual Processing registration, input validation, selection-aware
polygon/seed export, unsaved seed moves, output loading, individual step
parameters and defaults, setup construction, and the minimum-version
guard. Neural model calls are substituted at the engine boundary. The shared
Processing APIs can be checked with QGIS 3.44, but plugin startup requires 4.0.
The Qt6 menu lifecycle test runs only on QGIS 4+. The setup tests also cover
collapsed Advanced controls, install/repair status, asynchronous progress,
failure messages, closing during work, and removal of the selected engine.

For a conda QGIS installation whose Python bindings live under `share/qgis/python`:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH="$CONDA_PREFIX/share/qgis/python" \
  "$CONDA_PREFIX/bin/python" -m unittest discover -s tests -p 'test_qgis.py' -v
```

For optional real scientific dependency and raster validation checks, use a
separate supported Python environment with `samstars==0.2.0`:

```bash
/path/to/engine/bin/python -m unittest discover -s tests -p 'test_scientific.py' -v
CUDA_VISIBLE_DEVICES=-1 /path/to/engine/bin/python -m unittest discover -s tests -p 'test_steps.py' -v
```

`test_steps.py` runs four standard-library contract tests without an engine.
Its scientific cases cover real alignment across different grids/CRSs, NoData,
cost weighting and bands, seed controls and empty results, edited points, and
the proposal handoff. SAM inference alone is substituted. A tiny deterministic
StarDist network is saved and executed through the real refinement CLI on CPU,
including auxiliary normalization, window overlap, duplicate suppression,
GeoPackage output, and geometry repair that preserves crown identity and area.
This validates integration, not the accuracy of a trained crown model. The
tests create fixtures in temporary directories and do not alter engine packages.

No tests download checkpoints or run model training. Full model runs require
the user's imagery, CHM, density, annotations, and a ViT-L checkpoint or trained
bundle. Windows and macOS behavior require platform testing. GPU installation
pins the framework pair, TensorFlow, and NumPy; the complete dependency graph is
not a locked, reproducible scientific environment.

For the optional real GPU test, use an existing engine environment and explicitly
select the backend (`mps` on Apple Silicon):

```bash
SAMSTARS_TEST_GPU=cuda /path/to/engine/bin/python \
  -m unittest discover -s tests -p 'test_gpu.py' -v
```

This launches the same isolated worker used by setup. An unavailable GPU or a
failed computation fails the test. Omitting the environment variable skips it;
CPU-only CI does not claim GPU coverage.

Build the release artifact with `python3 scripts/package_plugin.py`; packaging
is tested for a valid `samstars_qgis/` root, required files, and QGIS 4.0 metadata.

Initial verification used Linux x64, QGIS 3.44.7 / Qt 5.15.13 / Python 3.12.3,
plus a disposable Python 3.10 environment with SAMStars 0.2.0, PyTorch 2.10.0,
TensorFlow 2.18.1, StarDist 0.9.1, and Rasterio 1.4.3. Real CPU and geospatial
operations passed. The existing conda geospatial packages had a CRS lookup
problem, so self-contained Rasterio, PyProj, and Pyogrio wheels were installed
only in the disposable environment.

The setup revision was verified on Linux x64 with **QGIS 4.2.3, Qt 6.11.2,
and Python 3.14.7**: all 20 QGIS integration tests passed. The separate 28
standard-library/runtime tests passed under Python 3.12. They cover installer
commands for Linux and macOS, automatic Python selection, owned-engine removal,
setup locking, and preservation of external/unmarked environments. Installer
package downloads are mocked in those tests. The QGIS startup logged missing
optional PDAL HDF reader libraries; the tested raster/vector workflows passed.

The GPU setup revision adds tests for platform/architecture gates, missing and
old drivers, hidden GPUs, CUDA build selection, native Python checks, constrained
install commands, and preservation of the previous engine after failed or
canceled GPU validation. QGIS tests cover the GPU dialog, diagnostic operations
that never change settings, repair retaining the GPU profile, and explicit CPU
selection. All package installation in these unit tests is mocked.

A real GPU diagnostic passed on Linux x64 with an **NVIDIA RTX 5090**, driver
**595.99.02**, Python **3.10**, **PyTorch 2.10.0+cu128**, and **Torchvision
0.25.0+cu128**. It executed matrix multiplication and convolution on CUDA and
verified gradients/results. This used an existing environment without changing
its packages. TensorFlow GPU behavior was not tested. The official cu126/cu128
indexes contain the pinned PyTorch and Torchvision wheels for Python 3.10–3.12.

The complete private CUDA 12.8 installer also passed on the same NVIDIA machine
in a fresh, disposable **Python 3.12.3** venv. This exercised real package
downloads, `pip check`, scientific imports, CPU operations, the compiled
StarDist extension, CRS transformation, GeoPackage I/O, and GPU operations with
gradients. It used TensorFlow 2.18.1, StarDist 0.9.2, NumPy 1.26.4, and Rasterio
1.4.4. The result explicitly reported CPU refinement. The test took about
22 minutes including downloads and unpacking. The temporary environment was
removed afterward; no QGIS engine selection or existing environment was changed.

The GPU revision's regression run passed **40 runtime tests, 25 QGIS 4/Qt6 tests, and the
optional real GPU test**. CUDA 12.6 installation on older hardware, Apple MPS,
and complete neural model training/inference remain unverified. A small GPU
diagnostic does not certify complete models, every GPU architecture, or every
QGIS 4.x release.

The Workflow steps revision adds four QGIS integration cases and eleven step
tests (four contracts and seven scientific cases). The independent tools use
the existing engine. Full SAM inference still requires testing with a trained
bundle; the SAM handoff test substitutes neural inference only.
The final run passed **40 runtime tests, 29 QGIS 4/Qt6 tests, and all 11 step
tests**. All five Processing forms also opened successfully in QGIS 4.2.3.
The real refinement check produced 25 synthetic crown polygons and verified
that the returned geometries were valid. This revision did not reinstall or
modify the existing engine or repeat the GPU installer test.

Before promoting GPU setup out of experimental status, validate on real release
targets (Linux NVIDIA cu126, Linux NVIDIA cu128, and Apple Silicon MPS):

- Fresh engine installation, CPU checks, GPU checks, and a small complete
  training/segmentation run with known inputs and expected outputs.
- CPU fallback, driver/missing-device errors, cancellation during download and
  validation, and preservation of the previously selected working engine.
- Each supported engine Python version, plus a full recorded dependency set
  and GPU/driver/OS details. Run CPU-only QGIS integration in CI; hardware tests
  require suitable machines. Expand the platform matrix only after these pass.
