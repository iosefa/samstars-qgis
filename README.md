# SAMStars for QGIS

Train models and segment individual tree crowns in the **QGIS Processing
Toolbox**. The plugin follows the PyForestScan QGIS plugin: ordinary Processing
dialogs, batch/model support, and one small setup dialog.

Requires **QGIS 4.0 or later**. Version **0.1.9 is experimental**.
Scientific libraries run in a separate Python
3.10–3.12 environment.

## Install

1. Build the plugin with `python3 scripts/package_plugin.py`. The result is
   `dist/samstars_qgis.zip`.
2. In QGIS, open **Plugins → Manage and Install Plugins → Install from ZIP**,
   select that ZIP, and enable **SAMStars**.
3. Open **Plugins → SAMStars → Set up SAMStars**, choose **Engine version**, and
   click **Install engine**. **Stable — 0.2.0** is the default;
   **Experimental — 0.3.0a1** opts into the alpha for testing.
   Setup detects an installed Python 3.10–3.12. Open **Advanced** to choose a
   Python executable yourself or select a prepared SAMStars environment with
   **Use existing environment**. Setup shows status and progress in the main
   dialog, with detailed logs under Advanced.
4. Run **Check SAMStars dependencies** in the Processing Toolbox. Its log and
   HTML report show import checks, small CPU operations, CRS transformation,
   and a GeoPackage round trip.

On macOS, automatic detection checks standard Homebrew and Python.org locations
even when QGIS is launched from Finder. If detection fails, open **Advanced** and
enter a compatible Python executable (for example,
`/opt/homebrew/bin/python3.12`), then click **Install engine**. **Use existing
environment** is for environments that already have SAMStars installed.

The private installer needs an existing Python with `venv`/`ensurepip`, internet
access, and several GB of disk space. It creates a new environment under the
active QGIS profile's `samstars/engines` directory, installs SAMStars and its
dependencies, and selects it only after validation. Linux defaults to CPU builds
of PyTorch/Torchvision. QGIS's Python and the selected base environment are not
modified. Failed or canceled installs remove only the new environment; older
engines remain available. **Repair engine** builds and validates a replacement
before selecting it. **Remove engine** removes only private environments marked
as belonging to this plugin; external environments and older unmarked installs
are preserved. A failed or canceled operation keeps its error visible in setup.
The base Python installation must remain available for its virtual environments.

Automatic installation targets **Linux x64** and **Apple Silicon macOS**,
following SAMStars's supported platforms (Ubuntu 20.04+ and macOS 14+).
Windows, Intel macOS, and Linux ARM are not tested SAMStars release targets;
they can select an existing environment, but this plugin does not promise
working scientific packages on those platforms. No portable Python, models,
checkpoints, or GPU runtimes are bundled. CPU setup pins SAMStars itself;
transitive dependencies resolve within its published constraints.

### Change engine version

Install the updated plugin ZIP and restart QGIS. In **Set up SAMStars**, choose
**Experimental — 0.3.0a1**, then click **Repair engine** (or **Install engine** if
no private engine is selected). Repair uses the selected version and preserves
the private engine's CPU/NVIDIA/Apple GPU profile. GPU setup also uses this version
choice. Installation creates a new environment and downloads its dependencies;
the current environment remains selected until validation succeeds, and is kept
afterwards. Changing the dropdown alone does not switch the active engine.

The setup summary shows the selected engine's version. To reuse a retained engine
without reinstalling, select its Python under **Advanced → Use existing environment**;
the plugin detects and validates either supported version. Failed or canceled
installs leave the previous selection in place. Both releases retain the same
Python 3.10–3.12 requirement and GPU setup policy.

## Optional GPU setup

Open **Set up SAMStars → GPU setup…** to install a GPU engine or **Test selected
engine** without changing it. GPU setup accelerates **SAM proposals and SAM
training (PyTorch)**. **StarDist refinement (TensorFlow) stays on CPU in private
engines**, including when CUDA or Apple MPS is selected. This distinction also
appears in the dialog and job log; it is not full-pipeline GPU acceleration.

| Computer | Automatic GPU setup |
| --- | --- |
| Linux x64, NVIDIA | CUDA PyTorch and its runtime dependencies, after a driver check |
| Apple Silicon, macOS 14+ | Native arm64 PyTorch with MPS; no separate GPU driver installation |
| Native Windows, Intel macOS, Linux ARM | Not offered; an existing environment can be selected and tested |
| AMD/Intel GPUs on Linux | Not offered; use CPU or a prepared environment |

For NVIDIA, setup queries `nvidia-smi`, uses CUDA 12.6 for older cards and 12.8
when a Blackwell-or-newer card is detected, and requires driver **560.35.05+** or
**570.124.06+**, respectively. These are deliberately conservative installer
requirements, not a claim about the minimum driver every CUDA application can
use. See the [NVIDIA driver tables](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-toolkit-release-notes/index.html)
and [PyTorch build pairs](https://pytorch.org/get-started/previous-versions/).
The final GPU computation must still pass: detecting a card cannot prove
that a wheel supports its kernels. Mixed GPU configurations and visibility
overrides are checked on the default GPU visible to the worker. Setup respects
`CUDA_VISIBLE_DEVICES`. It does not install drivers, change OS packages, enable
WSL, or compile a CUDA toolkit. WSL deployment is not a tested release target.

GPU profiles pin the PyTorch/Torchvision pair, TensorFlow, and NumPy; other
dependencies resolve within SAMStars's constraints. Installation requires binary
wheels, runs `pip check`, validates CPU/geospatial operations, then runs GPU
matrix/convolution operations and gradients in a separate process. Only a
successful installation becomes the selected engine. Failure or cancellation
removes the new environment and keeps the previous selection. **Repair engine**
preserves the selected private GPU profile. To return to CPU, choose **CPU
(default) → Install CPU engine** in GPU setup, or select a previous environment
under Advanced. Retained environments use additional disk space.

Select **CUDA** or **Apple MPS** in a Processing tool's Advanced parameters after
setup; CPU remains the default. Selecting a device never installs packages.
The GPU test checks small kernels, not full model compatibility or available
memory for a scene. Start with a small scene. GPU setup remains experimental
until complete training/inference runs pass on each supported release target.

TensorFlow's CUDA extras pin different NVIDIA libraries from the PyTorch
profiles, so setup does not combine them or bypass the dependency resolver.
It also does not install `tensorflow-metal`. Full TensorFlow GPU support needs
a separately validated dependency strategy; installing every optional GPU
package into the same environment is not supported by this installer.

For a custom CPU/GPU environment, follow the
[SAMStars installation guide](https://samstars.sefa.ai/installation/), then select
that environment's Python under **Advanced → Use existing environment**.
The [TensorFlow GPU guide](https://www.tensorflow.org/install/pip) and
[Apple Metal guide](https://developer.apple.com/metal/tensorflow-plugin/) describe
the optional runtime packages; keep their versions within SAMStars's dependency
constraints. For development with the neighboring source:

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
| Create training window | `samstars:createtrainingwindow` | Square polygon guide for labeling crowns |
| Align input rasters | `samstars:alignrasters` | Selected rasters aligned to a user-chosen reference grid |
| Build cost surface | `samstars:costsurface` | Boundary-cost raster |
| Generate seed points | `samstars:generateseeds` | Editable seed GeoPackage |
| Generate crown proposals (SAM) | `samstars:proposals` | Proposal records, probability, and canopy-index rasters |
| Refine crowns (StarDist) | `samstars:refine` | Crown GeoPackage from saved proposals |
| Check SAMStars dependencies | `samstars:checkdependencies` | HTML report and `READY` boolean |

**Tree crowns** contains the full segmentation and training workflows.
**Training preparation** contains the training-window guide tool.
**Workflow steps** contains the five independent steps. **Setup** contains
the dependency check. Segment tree crowns still handles alignment, cost
construction, seed generation, SAM proposals, and refinement automatically.

The full workflows take **imagery**, a **canopy height model in metres**, and a
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

To prepare hand-drawn labels, run **Training preparation → Create training
window**. Enter the square's side length in metres and choose a projected CRS
in metres, preferably the imagery CRS. Place it at the **centre of the current
map view**, or enter **X and Y in the window CRS**. Map-view coordinates are
transformed automatically. The default side length is 256 m; for example,
that covers 512 pixels per side at 0.5 m resolution.

Save the window to a GeoPackage to keep it. QGIS displays it as an outline so
you can create a separate polygon layer and trace one polygon per crown
inside it. Fully annotate the area and **crop the training rasters to it**
before training. The guide alone does not crop rasters, snap to pixels, or
set the model's SAM processing window. It needs no SAMStars engine.

To **train**, also select one polygon per labeled crown and a local
**SAM ViT-L** checkpoint (`sam_vit_l_0b3195.pth`, available from the
[official SAM checkpoint list](https://github.com/facebookresearch/segment-anything#model-checkpoints)).
Crop your rasters to a fully annotated area: unlabeled pixels are treated as
background. The plugin exports the requested polygon layer, including selected
features/subsets and unsaved edits, to a temporary GeoPackage. The output folder
is returned as `OUTPUT` and `MODEL`; pass it to segmentation's `MODEL` input.

**Train a model → Advanced** contains collapsible groups. Defaults preserve the
existing training behavior; optional overrides start disabled.

| Group | Settings |
| --- | --- |
| Input bands and execution | RGB, optional NIR/texture, device, intermediate storage |
| SAM windows and training | Pixel or metre windows, automatic or custom stride, epochs, minimum iterations, learning rate, extra foreground seeds |
| Seed generation | Peak and height thresholds, search radius, smoothing, minimum cluster samples, merge radius, thinning and distance |
| Proposal processing | Per-crop, mosaic, and post-fusion smoothing; batch prompt decoding |
| StarDist training | Epochs, minimum iterations, optional patch size, validation fraction, input channel, auxiliary CHM, normalization, augmentation |
| Defaults for prediction | Optional refinement windows/overlap and duplicate suppression, stored in the model bundle |

Choose **SAM window units** to enable either pixels (default **512**) or metres.
Automatic spacing is 75% of the pixel size, rounded down to at least one pixel,
retaining approximately 25% overlap. **Custom stride** enables a pixel distance
which must not exceed the window size. At 0.1 m resolution, 256 pixels covers
25.6 m; the default automatic stride is then 192 pixels. Custom pixel sizes,
metre windows, patch-size overrides, and extra foreground seed counts other
than 80 require **Experimental SAMStars 0.3.0a1**. Stable 0.2.0 requires 512-pixel
SAM windows, 80 extra foreground seeds, and the patch-size override off.
The experimental engine stores window/resolution settings for prediction
and pads images smaller than the window. Larger windows can need more memory.
SAM's fixed 1024-pixel encoder input is unchanged.

Set **Extra foreground seeds** directly under **SAM windows and training**.
The default is **80** additional foreground prompt points per seed; **0** keeps
only the seed as foreground. Background points and box prompts remain active.
**Override StarDist patch size** enables a separate
patch control, default 256; it must be a multiple of 16 and fit each training
image. Minimum iterations default to 0 (epochs only); positive values can
increase the effective epoch count. The default SAM learning rate is 0.00005.
Validation defaults to 0.2 and must be below 1. It splits proposal records,
not spatial areas; with one record, training data is reused for validation.

Thinning distance is enabled only when seed thinning is checked. Refinement
size/overlap are enabled only with tiled prediction, and overlap must be less
than the window size. Duplicate suppression independently enables an IoU
threshold between 0 and 1. These are saved prediction defaults, not training
patch settings. Inactive values are ignored and retained in the open dialog
when toggled off and back on. Batch decoding defaults to the engine's behavior
(off in 0.2.0, on in 0.3.0a1). Normalization also applies without auxiliary CHM.

During training, Processing displays SAM/StarDist epoch counts and StarDist
batch counts with an estimated time for the **current training stage** once
timing is available. This excludes later proposal generation, validation,
threshold optimization, and bundle saving as applicable. Terminal color codes
are removed from the log. A batch count such as `32/100` is not the overall
training percentage.

Advanced parameters also cover RGB band order, optional NIR/texture bands, compute
device, intermediate storage, and training epoch counts. RGB uses one-based
indices in red, green, blue order (training default: `1,2,3`). For segmentation,
blank RGB and zero NIR/texture reuse the bundle settings; zero does not disable
a band already stored in the model. CPU is the default. The device setting
selects PyTorch proposals. Private engines explicitly run refinement with CUDA
hidden, including in GPU profiles. In custom environments, TensorFlow manages
refinement acceleration independently; a custom TensorFlow Metal installation
can still select its own devices.

### Run individual workflow steps

Use **SAMStars → Workflow steps** to inspect results and adjust settings between
stages. Each tool opens its own Processing dialog and saves its outputs. Choose
a permanent output folder to keep the results. The first three steps need no
model bundle; proposal generation and refinement use a trained bundle.

1. **Align input rasters:** choose a **Reference raster**, then any of the four
   optional inputs: **Imagery**, **CHM**, **First-return density**, and **Intensity**.
   Select at least one input. The reference defines the CRS, extent, resolution,
   and pixel grid, and must be north-up with square pixels in a projected CRS
   in metres. For a CHM-based grid, select the CHM as reference and supply imagery,
   density, and intensity; leave the optional CHM input blank to reuse its source.
   Selected inputs produce `img.tif`, `chm.tif`, `den.tif`, or `intensity.tif`.
   Every imagery band is retained in its original order. Outputs are Float32;
   differing grids use bilinear resampling, while matching grids are copied
   without interpolation. `REFERENCE_RASTER` points to the original reference.
   Skip this step if the grids already match exactly.
2. **Build cost surface:** use imagery and CHM on the chosen reference grid
   (for example, the aligned imagery and original reference CHM). Set relative
   height-edge, NDVI-gap, and texture weights. Select RGB, NIR, and texture bands
   under Advanced to match training; NDVI is omitted without a NIR band.
   The result is `cost.tif`, with valid values from 0 to 1. You can save it in
   the input folder, provided `cost.tif` and `cost.workflow.json` do not already exist.
3. **Generate seed points:** use aligned CHM/density and the cost surface. Controls
   include the peak threshold (applied to both CHM and density), minimum canopy
   height, search radius, smoothing, and merge radius. Advanced exposes cost
   penalties, height splitting, and final thinning. Inspect `seeds.gpkg` in QGIS
   and move, add, or delete points as needed. You can use the same folder as
   the inputs and cost surface, provided `seeds.gpkg` and `seeds.workflow.json`
   do not already exist.
4. **Generate crown proposals (SAM):** select imagery and CHM on the reference grid,
   cost surface, edited seed layer, and trained model bundle. Selected features,
   layer filters, and unsaved edits are exported; seeds and costs are not
   regenerated. RGB defaults come from the bundle. Advanced offers CUDA/MPS,
   prompt batching, and smoothing controls. Probability and canopy-index rasters
   are loaded for inspection. Pass the **Proposal records** output
   (`proposals/proposal_records.json`) to the next step.
5. **Refine crowns (StarDist):** select those proposal records and the same model
   bundle. Leave thresholds at `-1` to use model settings, or override both
   together. Advanced offers window size/overlap, minimum crown area, edge
   exclusion, and duplicate suppression. This tool always uses CPU and reuses
   the saved proposals without rerunning SAM. It repairs invalid raster-derived
   outlines while preserving crown attributes, then loads `crowns/crowns.gpkg`.

The dialogs reject grid mismatches instead of silently resampling later inputs.
Proposals ignore seeds outside the image extent and report how many were skipped;
seeds on missing CHM/cost data must be corrected. Keep proposal folders and their
rasters in place: the records refer to those files. Step outputs and prepared
inputs are retained. Requested settings and outputs are recorded in
`cost.workflow.json` for costs, `seeds.workflow.json` for seeds, and
`workflow.json` for the other steps. Temporary Processing destinations still expire with the session.
Use new output folders to compare seed or refinement settings.

These tools operate on existing imagery/CHM/density rasters. They do not generate
CHM or density from raw point clouds. Model training remains the separate
**Train a model** workflow.

Cost and seed creation accept existing folders and protect their specific result
and settings files from being overwritten. Alignment, proposals, refinement,
training, and full segmentation require **new or empty output folders**, separate
from their inputs and model bundle. Intermediate files
in the full segmentation/training tools are temporary unless **Keep intermediate
files** is enabled; Workflow steps retain theirs. A canceled job
can leave partial outputs; choose a new folder for the next run. Temporary
Processing destinations last only for the session; choose a permanent folder
to retain a trained model or segmentation result.

Start with a small scene. Training and segmentation can require substantial
time, RAM, and disk space. Prepared-tile batch workflows and resume controls
remain in the SAMStars Python API. Cancellation stops the worker and its child
processes on the supported Linux/macOS platforms. Progress reports stage logs,
not an estimate of remaining time.

## Development and verification

```bash
python3 -m unittest discover -s tests -p 'test_runtime.py' -v
python3 -m unittest discover -s tests -p 'test_steps.py' -v
QT_QPA_PLATFORM=offscreen /usr/bin/python3 -m unittest discover -s tests -p 'test_qgis.py' -v
python3 scripts/package_plugin.py
```

Use a QGIS-enabled Python for the third command. The current integration tests
pass on **QGIS 4.2.3 / Qt 6.11.2**. They cover the setup dialog, background work
and cancellation, menu lifecycle, Processing parameters, registration, and
polygon/seed export, independent step dialogs, and training-window geometry and
placement. Scientific step tests also
run in a SAMStars environment. The plugin's startup guard rejects QGIS 3.x. A fresh private
CUDA 12.8 engine passed installation, CPU/geospatial checks, and real GPU
operations on an RTX 5090. **Full model training/inference and installation on
the other release targets still need testing before a stable release.**
See [tests/README.md](tests/README.md) for commands,
runtime verification details, and the optional real-engine and raster checks.

QGIS 4 metadata and Qt imports follow the
[official QGIS migration guide](https://github.com/qgis/QGIS/wiki/Plugin-migration-to-be-compatible-with-Qt5-and-Qt6).
Offline workflow help is included at [docs/index.html](docs/index.html).

## License

[GPL-3.0-or-later](LICENSE). The process client is adapted from
[PyForestScan for QGIS](https://github.com/iosefa/pyforestscan-qgis); see [NOTICE](NOTICE).
SAMStars and its dependencies are installed separately and retain their licenses.
