# SAMStars for QGIS

Train models and segment individual tree crowns in the **QGIS Processing
Toolbox**. The plugin follows the PyForestScan QGIS plugin: ordinary Processing
dialogs, batch/model support, and one small setup dialog.

Requires **QGIS 4.0 or later**. Version **0.1.0 is experimental** and targets the
SAMStars **0.2.0** public API. Scientific libraries run in a separate Python
3.10–3.12 environment.

## Install

1. Build the plugin with `python3 scripts/package_plugin.py`. The result is
   `dist/samstars_qgis.zip`.
2. In QGIS, open **Plugins → Manage and Install Plugins → Install from ZIP**,
   select that ZIP, and enable **SAMStars**.
3. Open **Plugins → SAMStars → Set up SAMStars**. Select a Python 3.10–3.12
   executable, then click **Install private engine**. Alternatively, select
   an environment that already contains `samstars==0.2.0` and click **Use existing**.
4. Run **Check SAMStars dependencies** in the Processing Toolbox. Its log and
   HTML report show import checks, small CPU operations, CRS transformation,
   and a GeoPackage round trip.

The private installer needs an existing Python with `venv`/`ensurepip`, internet
access, and several GB of disk space. It creates a new environment under the
active QGIS profile's `samstars/engines` directory, installs SAMStars and its
dependencies, and selects it only after validation. Linux installs CPU builds
of PyTorch/Torchvision. QGIS's Python and the selected base environment are not
modified. Failed or canceled installs remove only the new environment; older
engines remain available. Successful reinstallations retain older engines.
The base Python installation must remain available for its virtual environments.

Automatic installation targets **Linux x64** and **Apple Silicon macOS**,
following SAMStars's supported platforms (Ubuntu 20.04+ and macOS 14+).
Windows, Intel macOS, and Linux ARM are not tested SAMStars release targets;
they can select an existing environment, but this plugin does not promise
working scientific packages on those platforms. No portable Python, models,
checkpoints, or GPU runtimes are bundled. The installer pins SAMStars itself;
transitive dependencies resolve within its published constraints.

For a custom CPU/GPU environment, follow the
[SAMStars installation guide](https://samstars.sefa.ai/installation/), then select
that environment's Python in setup. For development with the neighboring source:

```bash
# Run inside a separate Python 3.10–3.12 environment.
python -m pip install -e ../sam
```

`SAMSTARS_PYTHON=/absolute/path/to/environment/bin/python` overrides the saved
selection, including for headless Processing. For a development symlink, use
the package name **samstars_qgis** under the active QGIS profile's
`python/plugins` directory; the repository's hyphenated name is not importable
as a QGIS plugin. Restart QGIS after updating the plugin.

## Use

| Processing tool | Algorithm ID | Result |
| --- | --- | --- |
| Segment tree crowns | `samstars:segment` | Crown GeoPackage, automatically loaded into the project |
| Train a model | `samstars:train` | Reusable model bundle folder |
| Check SAMStars dependencies | `samstars:checkdependencies` | HTML report and `READY` boolean |

Both workflows take **imagery**, a **canopy height model in metres**, and a
**first-return density raster**. Use local GDAL-readable rasters. Imagery must
have a projected CRS in metres and north-up square pixels. CHM and density
must each have one band, a defined CRS, and overlapping coverage. SAMStars
aligns them to the imagery grid; it cannot correct survey misregistration.
Supply full coverage and consistent density units/cell size for training and
inference. Raster pixel values are read directly; apply scale/offset metadata
before using CHM or density. Export web services and subdatasets to GeoTIFF.

To **segment**, choose a trained SAMStars bundle directory containing
`manifest.json`, then select a new output folder. The result is
`OUTPUT/crowns/crowns.gpkg`, also returned as the `CROWNS` Processing output.
Band choices come from the bundle unless overridden in Advanced parameters.

To **train**, also select one polygon per labeled crown and a local
**SAM ViT-L** checkpoint (`sam_vit_l_0b3195.pth`, available from the
[official SAM checkpoint list](https://github.com/facebookresearch/segment-anything#model-checkpoints)).
Crop your rasters to a fully annotated area: unlabeled pixels are treated as
background. The plugin exports the requested polygon layer, including selected
features/subsets and unsaved edits, to a temporary GeoPackage. The output folder
is returned as `OUTPUT` and `MODEL`; pass it to segmentation's `MODEL` input.

Advanced parameters cover RGB band order, optional NIR/texture bands, compute
device, intermediate storage, and training epoch counts. RGB uses one-based
indices in red, green, blue order (training default: `1,2,3`). For segmentation,
blank RGB and zero NIR/texture reuse the bundle settings; zero does not disable
a band already stored in the model. CPU is the default. The device setting
selects PyTorch proposals, and CPU mode hides CUDA from refinement.
TensorFlow manages refinement acceleration independently; a custom TensorFlow
Metal installation can still select its own devices.

Choose **new or empty output folders**. The plugin rejects reuse and output
paths that contain inputs or lie inside the model bundle. Intermediate files
are temporary unless **Keep intermediate files** is enabled. A canceled job
can leave partial outputs; choose a new folder for the next run. Temporary
Processing destinations last only for the session; choose a permanent folder
to retain a trained model or segmentation result.

Start with a small scene. Training and segmentation can require substantial
time, RAM, and disk space. The minimal plugin exposes the two single-scene
workflows; prepared tiles, resume controls, and individual pipeline stages
remain in the SAMStars Python API. Cancellation stops the worker and its child
processes on the supported Linux/macOS platforms. Progress reports stage logs,
not an estimate of remaining time.

## Development and verification

```bash
python3 -m unittest discover -s tests -p 'test_runtime.py' -v
QT_QPA_PLATFORM=offscreen /usr/bin/python3 -m unittest discover -s tests -p 'test_qgis.py' -v
python3 scripts/package_plugin.py
```

Use a QGIS-enabled Python for the second command. The tests also run against
QGIS 3.44's shared Processing APIs as a development check; the plugin's startup
guard still rejects QGIS 3.x. **QGIS 4/Qt6 runtime testing and a full model
training/inference run are required before a stable release.** Protocol,
cancellation, path protection, parameter mapping, real Processing registration,
and polygon export are covered separately. See [tests/README.md](tests/README.md)
for the optional real-engine and raster checks.

QGIS 4 metadata and Qt imports follow the
[official QGIS migration guide](https://github.com/qgis/QGIS/wiki/Plugin-migration-to-be-compatible-with-Qt5-and-Qt6).
Offline workflow help is included at [docs/index.html](docs/index.html).

## License

[GPL-3.0-or-later](LICENSE). The process client is adapted from
[PyForestScan for QGIS](https://github.com/iosefa/pyforestscan-qgis); see [NOTICE](NOTICE).
SAMStars and its dependencies are installed separately and retain their licenses.
