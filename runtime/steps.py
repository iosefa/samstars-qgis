"""Independent steps; scientific libraries are imported only when a step runs."""

from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


# These steps can share an existing folder. Reserve both the result and its
# settings record so one step never replaces another step's provenance.
SHARED_OUTPUT_FILES = {
    "costsurface": ("cost.tif", "cost.workflow.json"),
    "generateseeds": ("seeds.gpkg", "seeds.workflow.json"),
}


def number(value, name, *, minimum=0, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value < minimum or (maximum is not None and value > maximum) or (integer and int(value) != value):
        raise ValueError(f"Invalid {name}.")
    return int(value) if integer else float(value)


def _reference(dataset, *, image=False):
    if not dataset.crs or not dataset.crs.is_projected or not math.isclose(dataset.crs.linear_units_factor[1], 1):
        raise ValueError("The reference raster needs a projected CRS in metres.")
    t = dataset.transform
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0 or not math.isclose(t.a, -t.e, rel_tol=1e-6):
        raise ValueError("The reference raster must be north-up with square pixels.")
    if image and dataset.count < 3:
        raise ValueError("Imagery needs at least three RGB bands.")


def _single(dataset, label):
    if dataset.count != 1 or not dataset.crs:
        raise ValueError(f"{label} needs one band and a defined CRS.")
    if dataset.scales[0] != 1 or dataset.offsets[0] != 0:
        raise ValueError(f"Apply the {label} scale/offset before running this step.")
    if label == "chm" and dataset.units[0] and dataset.units[0].lower() not in ("m", "metre", "metres", "meter", "meters"):
        raise ValueError("Canopy height values must be in metres.")


def aligned(paths, *, image=False):
    """Reject grid mismatches rather than silently resampling edited inputs."""
    import rasterio

    items = list(paths.items())
    with ExitStack() as stack:
        datasets = [(label, stack.enter_context(rasterio.open(path))) for label, path in items]
        ref = datasets[0][1]
        _reference(ref, image=image)
        for index, (label, dataset) in enumerate(datasets):
            if not (index == 0 and image):
                _single(dataset, label)
            if (dataset.crs, dataset.transform, dataset.width, dataset.height) != (ref.crs, ref.transform, ref.width, ref.height):
                raise ValueError("Input rasters must share the same CRS, pixel grid, and extent. Run Align input rasters first.")
        return ref.crs, ref.bounds


def validate_cost(path):
    import numpy as np
    import rasterio

    found = False
    with rasterio.open(path) as dataset:
        for _, window in dataset.block_windows(1):
            values = dataset.read(1, window=window, masked=True).compressed()
            values = values[np.isfinite(values)]
            if values.size:
                found = True
                if values.min() < 0 or values.max() > 1:
                    raise ValueError("Cost values must be between 0 and 1, with missing pixels marked as NoData.")
    if not found:
        raise ValueError("The cost surface contains no valid pixels.")


def _bands(path, payload, defaults=None):
    import rasterio
    from samstars.features.cost_surface import _resolve_image_bands

    defaults = defaults or {}
    rgb = payload.get("rgb_bands") or defaults.get("rgb_bands") or [1, 2, 3]
    with rasterio.open(path) as dataset:
        return _resolve_image_bands(dataset.count, rgb, payload.get("nir_band"), payload.get("texture_band"))


def align_rasters(p, out, emit):
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    selected = [(key, output) for key, output in (
        ("img", "ALIGNED_IMAGE"), ("chm", "ALIGNED_CHM"),
        ("den", "ALIGNED_DENSITY"), ("intensity", "ALIGNED_INTENSITY"),
    ) if p.get(key)]
    if not selected:
        raise ValueError("Choose at least one raster to align to the reference.")
    result = {"REFERENCE_RASTER": str(Path(p["reference"]).resolve())}
    with rasterio.open(p["reference"]) as ref:
        _reference(ref)
        for index, (key, output) in enumerate(selected):
            with rasterio.open(p[key]) as source:
                if key == "img":
                    if not source.crs or source.count < 3:
                        raise ValueError("Imagery needs at least three bands and a defined CRS.")
                else:
                    _single(source, key)
                bounds = transform_bounds(source.crs, ref.crs, *source.bounds)
                if bounds[0] >= ref.bounds.right or bounds[2] <= ref.bounds.left or bounds[1] >= ref.bounds.top or bounds[3] <= ref.bounds.bottom:
                    raise ValueError(f"{key} does not overlap the reference raster.")
                path = out / f"{key}.tif"
                profile = dict(driver="GTiff", width=ref.width, height=ref.height, count=source.count,
                               crs=ref.crs, transform=ref.transform, dtype="float32", nodata=-9999.0,
                               tiled=True, compress="deflate", BIGTIFF="IF_SAFER")
                same_grid = (source.crs, source.transform, source.width, source.height) == (
                    ref.crs, ref.transform, ref.width, ref.height,
                )
                with ExitStack() as stack:
                    reader = source if same_grid else stack.enter_context(WarpedVRT(
                        source, crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height,
                        dtype="float32", nodata=-9999.0, resampling=Resampling.bilinear,
                    ))
                    dest = stack.enter_context(rasterio.open(path, "w", **profile))
                    for _, window in dest.block_windows(1):
                        for band in range(1, source.count + 1):
                            data = reader.read(band, window=window, masked=True).astype("float32").filled(-9999.0)
                            data[~np.isfinite(data)] = -9999.0
                            dest.write(data, band, window=window)
                    dest.colorinterp = source.colorinterp
                    dest.scales, dest.offsets = source.scales, source.offsets
                    for band in range(1, source.count + 1):
                        if source.descriptions[band - 1]:
                            dest.set_band_description(band, source.descriptions[band - 1])
                        if source.units[band - 1]:
                            dest.set_band_unit(band, source.units[band - 1])
                    if key == "chm":
                        dest.set_band_unit(1, "m")
                result[output] = str(path)
            emit(f"Aligned {key} to the reference grid.", 10 + 80 * (index + 1) / len(selected))
    return result


def cost_surface(p, out, emit):
    from samstars.features import build_cost_surface, write_cost_surface

    aligned({"img": p["img"], "chm": p["chm"]}, image=True)
    rgb = _bands(p["img"], p)
    weights = p.get("weights", [.4, .3, .3])
    if not isinstance(weights, list) or len(weights) != 3:
        raise ValueError("Supply three cost weights: height, NDVI, and texture.")
    weights = [number(value, "cost weight") for value in weights]
    if not sum(weights) or weights[0] + weights[2] + (weights[1] if p.get("nir_band") else 0) <= 0:
        raise ValueError("Give a positive weight to an available cost term; NDVI needs a selected NIR band.")
    weights = tuple(value / sum(weights) for value in weights) + (0.0,)
    emit("Building boundary costs…", 20)
    cost, profile = build_cost_surface(
        img_path=p["img"], chm_path=p["chm"], rgb_bands=rgb,
        nir_band=p.get("nir_band"), texture_band=p.get("texture_band"), weights=weights,
    )
    path = write_cost_surface(cost, profile, out / "cost.tif", exist_ok=False)
    validate_cost(path)
    return {"COST": str(path)}


def generate_seeds(p, out, emit):
    from samstars.features import SeedParams, canonical_seeds
    from samstars.data.tiling import _spatial_thin

    aligned({key: p[key] for key in ("chm", "den", "cost")})
    validate_cost(p["cost"])
    allowed = {"d_min", "min_canopy_height", "min_dist_px", "gauss_sigma", "merge_radius", "cost_weight", "dz_merge"}
    settings = p.get("seed_settings", {})
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError("Unknown seed settings.")
    settings = {key: number(value, key, minimum=.001 if key == "merge_radius" else 1 if key == "min_dist_px" else 0,
                            integer=key in ("min_dist_px", "gauss_sigma")) for key, value in settings.items()}
    thin = number(p.get("thin_distance", .8), "seed thinning distance")
    emit("Detecting and grouping seed candidates…", 20)
    try:
        seeds = canonical_seeds(SeedParams(chm_raster=Path(p["chm"]), den_raster=Path(p["den"]),
                                          cost_raster=Path(p["cost"]), **settings))
    except SystemExit as exc:
        raise ValueError(f"No seed points were found. Check the raster values and seed thresholds. {exc}") from exc
    if thin:
        seeds = _spatial_thin(seeds, thin)
    if seeds.empty:
        raise ValueError("No seed points remain. Check the seed thresholds and thinning distance.")
    seeds = seeds.reset_index(drop=True)
    seeds["id"] = range(len(seeds))
    path = out / "seeds.gpkg"
    seeds.to_file(path, driver="GPKG", layer="seeds", engine="pyogrio", index=False)
    emit(f"Saved {len(seeds)} editable seed points.", 90)
    return {"SEEDS": str(path)}


def _model(path):
    from samstars import SamStars

    model = SamStars.load(Path(path))
    if model.manifest.get("bundle_version") != 1 or not isinstance(model.defaults, dict):
        raise ValueError("Unsupported model bundle. Choose a trained SAMStars bundle.")
    return model


def _copy_rgb(source, destination, bands):
    import rasterio
    from rasterio.enums import ColorInterp

    with rasterio.open(source) as src:
        profile = dict(driver="GTiff", width=src.width, height=src.height, count=3,
                       dtype=src.dtypes[0], crs=src.crs, transform=src.transform, nodata=src.nodata,
                       compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(destination, "w", **profile) as dst:
            for _, window in dst.block_windows(1):
                dst.write(src.read(bands, window=window), window=window)
                dst.write_mask(src.dataset_mask(window=window), window=window)
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)


def _copy_single(source, destination):
    """Materialize GDAL inputs and their masks as a standalone GeoTIFF."""
    import numpy as np
    import rasterio

    with rasterio.open(source) as src:
        profile = dict(driver="GTiff", width=src.width, height=src.height, count=1,
                       dtype="float32", crs=src.crs, transform=src.transform,
                       nodata=-9999.0, compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
        with rasterio.open(destination, "w", **profile) as dst:
            for _, window in dst.block_windows(1):
                data = src.read(1, window=window, masked=True).astype("float32").filled(-9999.0)
                data[~np.isfinite(data)] = -9999.0
                dst.write(data, 1, window=window)


def _prepare_points(path, crs, bounds, chm, cost, emit):
    import geopandas as gpd
    import numpy as np
    import rasterio

    seeds = gpd.read_file(path, engine="pyogrio")
    if seeds.empty or seeds.crs is None:
        raise ValueError("Choose at least one seed point with a defined CRS.")
    if seeds.geometry.isna().any() or seeds.geometry.is_empty.any() or not seeds.geometry.is_valid.all():
        raise ValueError("Seed points contain empty or invalid geometry.")
    seeds = seeds.explode(index_parts=False).reset_index(drop=True)
    if not seeds.geometry.geom_type.eq("Point").all():
        raise ValueError("The seed layer must contain points.")
    seeds = seeds.to_crs(crs)
    x, y = seeds.geometry.x, seeds.geometry.y
    inside = np.isfinite(x) & np.isfinite(y) & (x >= bounds.left) & (x < bounds.right) & (y > bounds.bottom) & (y <= bounds.top)
    count = int((~inside).sum())
    seeds = seeds[inside].copy().reset_index(drop=True)
    if count:
        emit(f"Ignored {count} seed points outside the imagery extent.")
    if seeds.empty:
        raise ValueError("No seed points fall inside the imagery extent.")
    coordinates = list(zip(seeds.geometry.x, seeds.geometry.y))
    for raster in (chm, cost):
        with rasterio.open(raster) as dataset:
            for value in dataset.sample(coordinates, masked=True):
                if np.ma.is_masked(value) or not np.isfinite(value).all():
                    raise ValueError("A seed falls on missing CHM or cost data. Move/remove that seed or correct the rasters.")
    # Only geometry and fresh IDs are needed; editable user attributes stay in
    # the original layer, which is never modified.
    return gpd.GeoDataFrame({"id": range(len(seeds))}, geometry=seeds.geometry, crs=crs)


def proposals(p, out, emit):
    import torch
    from samstars.data import build_tile_records
    from samstars.segmentation.proposals import run_tile_proposals

    crs, bounds = aligned({key: p[key] for key in ("img", "chm", "cost")}, image=True)
    validate_cost(p["cost"])
    model = _model(p["model"])
    rgb = _bands(p["img"], p, model.defaults)
    device = p.get("device", "cpu")
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Choose CPU, CUDA, or Apple MPS.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use CPU or check GPU setup.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is unavailable. Use CPU or check GPU setup.")
    smoothing = {key: number(p.get(key, model.defaults.get(key, default)), key) for key, default in (
        ("crop_smooth_sigma", 2), ("smooth_sigma", 4), ("post_smooth_sigma", 0),
    )}
    seeds = _prepare_points(p["seeds"], crs, bounds, p["chm"], p["cost"], emit)
    tile = out / "prepared" / "tile_scene"
    tile.mkdir(parents=True)
    emit("Saving the selected RGB bands and edited seeds…", 10)
    _copy_rgb(p["img"], tile / "img.tif", rgb)
    _copy_single(p["chm"], tile / "chm.tif")
    _copy_single(p["cost"], tile / "cost.tif")
    seeds.to_file(tile / "seeds.gpkg", layer="seeds", driver="GPKG", engine="pyogrio", index=False)
    _, records = build_tile_records(tiles_dir=tile.parent, out_root=out / "run_records")
    emit(f"Running SAM proposals with {len(seeds)} seed points on {device}…", 25)
    proposal_root = out / "proposals"
    options = dict(smoothing)
    preprocessing = getattr(model, "preprocessing", None)
    if preprocessing is not None:
        options["preprocessing"] = preprocessing
    run_tile_proposals(records, proposal_root, model.decoder_path, device=device, fast=bool(p.get("fast", True)), **options)
    records_path = proposal_root / "proposal_records.json"
    records = read_proposal_records(records_path)
    if len(records) != 1:
        raise RuntimeError("The single-scene proposal step did not produce one complete proposal record.")
    return {"PROPOSAL_RECORDS": str(records_path), "PROBABILITY": records[0]["file_name"], "CANOPY_INDEX": records[0]["index"]}


RASTER_FIELDS = ("file_name", "chm", "cost", "logit", "index")


def read_proposal_records(path):
    path = Path(path).expanduser().resolve()
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Could not read the proposal records JSON.") from exc
    if not isinstance(records, list) or not records:
        raise ValueError("Proposal records must contain at least one completed proposal raster set.")
    result = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid proposal record.")
        normalized = {}
        for key in RASTER_FIELDS:
            value = record.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Proposal record is missing {key}.")
            raster = Path(value).expanduser()
            raster = (raster if raster.is_absolute() else path.parent / raster).resolve()
            if not raster.is_file():
                raise ValueError(f"Missing proposal raster: {raster}. Keep the proposal folder with its records.")
            normalized[key] = str(raster)
        # Metadata is optional for refinement; derive CRS/grid from the rasters.
        normalized.update(mask="", meta={})
        result.append(normalized)
    return result


def refinement_settings(p, defaults):
    if ("stardist_prob_thr" in p) != ("stardist_nms_thr" in p):
        raise ValueError("Override both refinement thresholds together.")
    values = {key: p.get(key, defaults.get(key, default)) for key, default in (
        ("stardist_prob_thr", None), ("stardist_nms_thr", None),
        ("stardist_window_size_m", None), ("stardist_window_overlap_m", 0),
        ("stardist_dedup_iou_thr", 0),
    )}
    if (values["stardist_prob_thr"] is None) != (values["stardist_nms_thr"] is None):
        raise ValueError("The model has incomplete thresholds. Override both thresholds explicitly.")
    for key in ("stardist_prob_thr", "stardist_nms_thr"):
        if values[key] is not None:
            values[key] = number(values[key], key, maximum=1)
    size = values["stardist_window_size_m"]
    values["stardist_window_size_m"] = number(size, "window size") if size is not None else 0
    values["stardist_window_overlap_m"] = number(values["stardist_window_overlap_m"], "window overlap")
    size, overlap = values["stardist_window_size_m"], values["stardist_window_overlap_m"]
    if overlap and (not size or overlap >= size):
        raise ValueError("Window overlap must be smaller than a positive refinement window size.")
    dedup = number(values["stardist_dedup_iou_thr"], "duplicate crown overlap threshold", maximum=1)
    if dedup == 1:
        raise ValueError("The duplicate crown overlap threshold must be below 1.")
    values["stardist_dedup_iou_thr"] = dedup
    values["min_area"] = number(p.get("min_area", 0), "minimum crown area")
    values["edge_buffer"] = number(p.get("edge_buffer", .5), "edge buffer")
    return values


def _finish_crowns(path, emit):
    """Repair diagonal pixel contacts without splitting a crown's attributes."""
    import geopandas as gpd
    from shapely import make_valid
    from shapely.geometry import MultiPolygon, Polygon

    crowns = gpd.read_file(path, layer="crowns", engine="pyogrio")
    if crowns.crs is None:
        raise RuntimeError("The refined crowns have no CRS.")
    if crowns.geometry.isna().any() or crowns.geometry.is_empty.any():
        raise RuntimeError("Refinement returned empty crown geometry. Inspect the processing log.")
    invalid = ~crowns.geometry.is_valid
    if invalid.any():
        def polygons(geometry):
            if isinstance(geometry, Polygon):
                return [geometry] if not geometry.is_empty else []
            return [part for child in getattr(geometry, "geoms", ()) for part in polygons(child)]

        for index in crowns.index[invalid]:
            parts = polygons(make_valid(crowns.geometry.loc[index]))
            if not parts:
                raise RuntimeError("A refined crown has no polygon area after geometry repair.")
            crowns.loc[index, crowns.geometry.name] = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        if not crowns.geometry.is_valid.all():
            raise RuntimeError("Crown geometry repair failed. Inspect the saved crown layer.")
        temporary = path.with_name("crowns.validated.gpkg")
        crowns.to_file(temporary, driver="GPKG", layer="crowns", engine="pyogrio",
                       geometry_type="MultiPolygon", promote_to_multi=True, index=False)
        temporary.replace(path)
        emit(f"Repaired {int(invalid.sum())} crown geometries from raster boundaries; crown attributes were preserved.")
    emit(f"Saved {len(crowns)} crown polygons.", 95)


def refine(p, out, emit):
    import samstars
    from samstars.data import build_scale_stats_from_proposal_records

    model = _model(p["model"])
    settings = refinement_settings(p, model.defaults)
    records = p["_records"]
    input_key = model.defaults.get("stardist_input_key", "index")
    if input_key not in RASTER_FIELDS:
        raise ValueError("The model requests an unsupported proposal input channel.")
    reference_crs, names = None, set()
    for record in records:
        crs, _ = aligned({key: record[key] for key in RASTER_FIELDS})
        if reference_crs is not None and crs != reference_crs:
            raise ValueError("All proposal records must use the same CRS for a combined crown layer.")
        reference_crs = crs
        name = Path(record[input_key]).stem
        if name in names:
            raise ValueError("Proposal records have duplicate raster names. Refine those scenes separately.")
        names.add(name)
    records_path = out / "proposal_records.json"
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    use_aux = bool(model.defaults.get("stardist_use_aux", True))
    scale_stats = out / "proposal_scale_stats.json"
    if use_aux:
        emit("Preparing refinement normalization statistics…", 10)
        build_scale_stats_from_proposal_records(proposal_records=records_path, out=scale_stats,
                                               percentiles=(2.0, 98.0), chm_key="chm", nodata_values=(-9999.0,))
    with tempfile.TemporaryDirectory(prefix="samstars-refine-model-") as directory:
        staged = model._stage_stardist_model(Path(directory))
        script = Path(samstars.__file__).parent / "segmentation" / "refinement.py"
        # Hide all TF GPU devices, including optional Metal devices in external
        # engines. The subprocess inherits the worker's cancellation group.
        bootstrap = (
            "import runpy,sys; import tensorflow as tf; tf.config.set_visible_devices([], 'GPU'); "
            "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        command = [sys.executable, "-I", "-c", bootstrap, str(script),
                   "--proposal-records", str(records_path), "--model-dir", str(staged.parent),
                   "--model-name", staged.name, "--out-dir", str(out / "crowns"),
                   "--input-key", input_key, "--preds-name", "crowns.gpkg", "--log",
                   "--min-area-m2", str(settings["min_area"]), "--edge-buffer-px", str(settings["edge_buffer"]),
                   "--dedup-iou-thr", str(settings["stardist_dedup_iou_thr"])]
        if use_aux:
            command += ["--use-aux", "--scale-stats", str(scale_stats)]
        if settings["stardist_prob_thr"] is not None:
            command += ["--prob-thr", str(settings["stardist_prob_thr"]), "--nms-thr", str(settings["stardist_nms_thr"])]
        if settings["stardist_window_size_m"]:
            command += ["--window-size-m", str(settings["stardist_window_size_m"]),
                        "--window-overlap-m", str(settings["stardist_window_overlap_m"])]
        emit("Refining saved proposals with StarDist on CPU…", 25)
        try:
            subprocess.run(command, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("StarDist refinement failed. See the processing log for the underlying error.") from exc
    crowns = out / "crowns" / "crowns.gpkg"
    if not crowns.is_file():
        raise RuntimeError("Refinement produced no crown GeoPackage. Check the thresholds and proposal coverage.")
    _finish_crowns(crowns, emit)
    return {"CROWNS": str(crowns)}


def run_step(operation, payload, *, emit, validate_output, claim_output):
    p = dict(payload)
    keys = {
        "alignrasters": ("reference",), "costsurface": ("img", "chm"),
        "generateseeds": ("chm", "den", "cost"),
        "proposals": ("img", "chm", "cost", "seeds", "model"), "refine": ("records", "model"),
    }[operation]
    inputs = [p[key] for key in keys]
    if operation == "alignrasters":
        inputs += [p[key] for key in ("img", "chm", "den", "intensity") if p.get(key)]
    if operation == "refine":
        p["_records"] = read_proposal_records(p["records"])
        inputs += [record[key] for record in p["_records"] for key in RASTER_FIELDS]
    filenames = SHARED_OUTPUT_FILES.get(operation)
    out = validate_output(p["out"], inputs, filenames=filenames)
    functions = {"alignrasters": align_rasters, "costsurface": cost_surface, "generateseeds": generate_seeds,
                 "proposals": proposals, "refine": refine}
    with claim_output(out, filenames=filenames):
        result = {"OUTPUT": str(out), **functions[operation](p, out, emit)}
        record = filenames[-1] if filenames else "workflow.json"
        with (out / record).open("x", encoding="utf-8") as stream:
            json.dump({"step": operation, "parameters": payload, "outputs": result}, stream, indent=2, allow_nan=False)
    emit("Workflow step finished.", 100)
    return result
