# Dry forest example rasters

A **50 m × 50 m** square centered at **x = 202591, y = 2112546** in
**EPSG:6635** (NAD83(PA11) / UTM zone 5N). The extent reaches 25 m from the
center in each direction.

| Property | Value |
| --- | --- |
| West / east | 202566 / 202616 m |
| South / north | 2112521 / 2112571 m |
| Dimensions | 100 × 100 pixels per raster |
| Pixel size | 0.5 × 0.5 m |
| Data type | Float32 |
| NoData marker | -9999 |
| Valid coverage | 100% in every band |

| File | Bands | Input |
| --- | --- | --- |
| [chm.tif](chm.tif) | 1 | Canopy height model in metres; reference grid |
| [multispectral.tif](multispectral.tif) | 8 | Multispectral imagery, original band order |
| [density.tif](density.tif) | 1 | First-return density |
| [intensity.tif](intensity.tif) | 1 | LiDAR intensity |

The four GeoTIFFs total about **315 KiB**, using lossless DEFLATE compression.
They share the same CRS, extent, transform, and dimensions and are ready to
load into QGIS. No alignment step is needed before using them together.

For **Build cost surface**, select `multispectral.tif` as Imagery and `chm.tif`
as CHM. For **Generate seed points**, use `chm.tif`, `density.tif`, and the
resulting cost raster. The imagery retains all eight bands without assigning
spectral names; select RGB and optional NIR bands according to the source band
mapping. Segmentation and crown proposals still require a trained model bundle;
training also requires crown annotations and a SAM checkpoint.

Save generated results in a separate working folder, such as the checkout's
ignored `local_data/` directory. These example inputs are tracked in Git and
are available with the repository; they are not bundled in the plugin ZIP.

## Provenance

The examples are exact pixel-window copies of the previously prepared 1 km
test inputs, using the CHM grid:

| Example | Source relative to the repository root |
| --- | --- |
| `chm.tif` | `local_data/epsg6635_x202591_y2112546_1km/chm.tif` |
| `multispectral.tif` | `local_data/aligned_to_chm/img.tif` |
| `density.tif` | `local_data/aligned_to_chm/den.tif` |
| `intensity.tif` | `local_data/aligned_to_chm/intensity.tif` |

Each crop starts at zero-based column 950 and row 950 and reads 100 columns
and 100 rows. Cropping applies no resampling, normalization, or band selection.
The larger aligned imagery had already been bilinearly resampled to the CHM
grid. All crop values and masks were checked against the corresponding source
windows, and the original files were retained unchanged. The larger sources
remain local and Git-ignored; they are not required to use these examples.
