"""Native Processing definitions and conditional training parameter mapping."""

from qgis.core import (
    QgsProcessingException, QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum, QgsProcessingParameterNumber,
)

from ..runtime.training import training_options

GROUPS = {
    "inputs": "Input bands and execution",
    "sam": "SAM windows and training",
    "seeds": "Seed generation",
    "proposals": "Proposal processing",
    "stardist": "StarDist training",
    "prediction": "Defaults for prediction",
}

ORDER = (
    "SAM_WINDOW_UNITS", "SAM_WINDOW_SIZE", "SAM_WINDOW_METRES", "SAM_STRIDE_MODE", "SAM_STRIDE",
    "SAM_EPOCHS", "SAM_MIN_ITERS", "SAM_LR", "FG_EXTRA",
    "SEED_D_MIN", "SEED_MIN_HEIGHT", "SEED_MIN_DIST", "SEED_GAUSS", "SEED_MIN_SAMPLES",
    "SEED_MERGE_RADIUS", "SEED_THINNING", "SEED_THIN_DISTANCE",
    "CROP_SMOOTHING", "MOSAIC_SMOOTHING", "POST_SMOOTHING", "FAST_PROPOSALS",
    "STARDIST_EPOCHS", "STARDIST_MIN_ITERS", "OVERRIDE_PATCH_SIZE", "STARDIST_PATCH_SIZE",
    "VALIDATION_FRACTION", "STARDIST_INPUT", "STARDIST_USE_AUX", "STARDIST_NORMALIZE", "STARDIST_AUGMENT",
    "REFINEMENT_TILING", "STARDIST_WINDOW_SIZE", "STARDIST_OVERLAP", "REFINEMENT_DEDUP", "STARDIST_DEDUP",
)

# Name, API keyword, label, default, minimum, integer, group.
NUMBERS = (
    ("SAM_WINDOW_SIZE", "chip_size", "SAM training window size (pixels)", 512, 1, True, "sam"),
    ("SAM_WINDOW_METRES", "sam_window_size_m", "SAM training window size (metres)", 50, .000001, False, "sam"),
    ("SAM_STRIDE", "chip_stride", "SAM window stride (pixels)", 384, 1, True, "sam"),
    ("SAM_EPOCHS", "sam_epochs", "SAM training epochs", 15, 1, True, "sam"),
    ("SAM_MIN_ITERS", "sam_min_iters", "Minimum SAM iterations (0 = epochs only)", 0, 0, True, "sam"),
    ("SAM_LR", "sam_lr", "SAM learning rate", .00005, .000000001, False, "sam"),
    ("FG_EXTRA", "fg_extra", "Extra foreground seeds", 80, 0, True, "sam"),
    ("SEED_D_MIN", "seed_d_min", "Minimum peak value (CHM and density)", 4.5, 0, False, "seeds"),
    ("SEED_MIN_DIST", "seed_min_dist_px", "Peak-search radius (pixels)", 4, 1, True, "seeds"),
    ("SEED_GAUSS", "seed_gauss_sigma", "Seed smoothing (pixels, 0 = disabled)", 2, 0, True, "seeds"),
    ("SEED_MIN_SAMPLES", "seed_min_samples", "Minimum samples per seed cluster", 1, 1, True, "seeds"),
    ("SEED_MERGE_RADIUS", "seed_merge_radius", "Seed merge radius (metres)", 3, .000001, False, "seeds"),
    ("SEED_MIN_HEIGHT", "seed_min_canopy_height", "Minimum canopy height (metres)", 2, 0, False, "seeds"),
    ("SEED_THIN_DISTANCE", "thin_dist", "Seed thinning distance (metres)", .8, 0, False, "seeds"),
    ("CROP_SMOOTHING", "crop_smooth_sigma", "Per-crop smoothing (pixels, 0 = disabled)", 2, 0, False, "proposals"),
    ("MOSAIC_SMOOTHING", "smooth_sigma", "Mosaic smoothing (pixels, 0 = disabled)", 4, 0, False, "proposals"),
    ("POST_SMOOTHING", "post_smooth_sigma", "Post-fusion smoothing (pixels, 0 = disabled)", 0, 0, False, "proposals"),
    ("STARDIST_EPOCHS", "stardist_epochs", "StarDist training epochs", 50, 1, True, "stardist"),
    ("STARDIST_MIN_ITERS", "stardist_min_iters", "Minimum StarDist iterations (0 = epochs only)", 0, 0, True, "stardist"),
    ("STARDIST_PATCH_SIZE", "stardist_patch_size", "StarDist training patch size (pixels, multiple of 16)", 256, 16, True, "stardist"),
    ("VALIDATION_FRACTION", "val_fraction", "Validation fraction (0 = no held-out records)", .2, 0, False, "stardist"),
    ("STARDIST_WINDOW_SIZE", "stardist_window_size_m", "Refinement window size (metres)", 50, .000001, False, "prediction"),
    ("STARDIST_OVERLAP", "stardist_window_overlap_m", "Refinement window overlap (metres)", 0, 0, False, "prediction"),
    ("STARDIST_DEDUP", "stardist_dedup_iou_thr", "Duplicate crown IoU threshold (between 0 and 1)", .5, .000001, False, "prediction"),
)

BOOLEANS = (
    ("SEED_THINNING", "thin_seeds", "Thin nearby seeds", True, "seeds"),
    ("OVERRIDE_PATCH_SIZE", None, "Override StarDist patch size (SAMStars 0.3.0a1)", False, "stardist"),
    ("STARDIST_USE_AUX", "stardist_use_aux", "Use CHM as an auxiliary StarDist input", True, "stardist"),
    ("STARDIST_NORMALIZE", "stardist_normalize_inputs", "Normalize StarDist input channels", True, "stardist"),
    ("STARDIST_AUGMENT", "stardist_augment", "Augment StarDist training samples", True, "stardist"),
    ("REFINEMENT_TILING", None, "Use tiled refinement for prediction", False, "prediction"),
    ("REFINEMENT_DEDUP", None, "Suppress duplicate crowns during prediction", False, "prediction"),
)

ENUMS = (
    ("SAM_WINDOW_UNITS", "SAM window units", ["Pixels", "Metres (SAMStars 0.3.0a1)"], "sam"),
    ("SAM_STRIDE_MODE", "SAM window spacing", ["Automatic (25% overlap)", "Custom stride in pixels"], "sam"),
    ("FAST_PROPOSALS", "Batch prompt decoding", ["Engine default", "Enabled", "Disabled"], "proposals"),
    ("STARDIST_INPUT", "StarDist input", ["Canopy index", "SAM probability"], "stardist"),
)

DEPENDENCIES = {
    "SAM_WINDOW_SIZE": ("SAM_WINDOW_UNITS", 0),
    "SAM_WINDOW_METRES": ("SAM_WINDOW_UNITS", 1),
    "SAM_STRIDE": ("SAM_STRIDE_MODE", 1),
    "SEED_THIN_DISTANCE": ("SEED_THINNING", True),
    "STARDIST_PATCH_SIZE": ("OVERRIDE_PATCH_SIZE", True),
    "STARDIST_WINDOW_SIZE": ("REFINEMENT_TILING", True),
    "STARDIST_OVERLAP": ("REFINEMENT_TILING", True),
    "STARDIST_DEDUP": ("REFINEMENT_DEDUP", True),
}

HELP = {
    "SAM_WINDOW_SIZE": "Square window in native pixels. Custom sizes require SAMStars 0.3.0a1 and are stored for prediction. SAM still resizes each window to its fixed 1024-pixel encoder input.",
    "SAM_WINDOW_METRES": "Square ground footprint in metres, rounded to the nearest native pixel. Requires SAMStars 0.3.0a1. Only the selected unit is used.",
    "SAM_STRIDE": "Distance between window starts in native pixels. Must not exceed the window width. Automatic spacing uses 75% of the window width, rounded down to at least one pixel.",
    "SAM_MIN_ITERS": "Optional minimum iteration budget; the engine can increase the epoch count to reach it. Zero uses the epoch count alone.",
    "STARDIST_MIN_ITERS": "Optional minimum iteration budget interpreted by the engine; may increase the epoch count. Zero uses epochs alone.",
    "FG_EXTRA": "Maximum additional foreground prompt points per seed (default 80). Set 0 to keep only the seed as foreground. Background points and box prompts are unchanged. Values other than 80 require SAMStars 0.3.0a1.",
    "STARDIST_PATCH_SIZE": "Independent of the SAM window. Must be a positive multiple of 16 and fit every training/validation image. Leave the override off to use the engine default (256 pixels).",
    "VALIDATION_FRACTION": "Fraction of proposal records reserved for validation, from 0 up to but excluding 1. This is not a spatial holdout. With a single scene/record, the engine reuses training data for validation.",
    "STARDIST_NORMALIZE": "Normalize each refinement input channel. Applies to the primary channel even when auxiliary CHM is disabled.",
    "FAST_PROPOSALS": "Engine default preserves existing behavior: enabled in 0.3.0a1, disabled in 0.2.0. Batches prompt decoding and does not change the chosen window size in 0.3.0a1.",
    "STARDIST_OVERLAP": "Used only for tiled refinement during later prediction. Must be smaller than the refinement window size.",
    "STARDIST_DEDUP": "Saved for later prediction. Positive IoU threshold below 1; lower values suppress more overlapping crowns. Independent of whether refinement uses windows.",
}


def add_training_parameters(algorithm):
    definitions = {}
    for name, label, options, group in ENUMS:
        definitions[name] = (QgsProcessingParameterEnum(name, label, options=options, defaultValue=0), group)
    for name, _, label, default, group in BOOLEANS:
        definitions[name] = (QgsProcessingParameterBoolean(name, label, defaultValue=default), group)
    for name, _, label, default, minimum, integer, group in NUMBERS:
        kwargs = {"maxValue": 1} if name in ("VALIDATION_FRACTION", "STARDIST_DEDUP") else {}
        parameter = QgsProcessingParameterNumber(
            name, label, type=QgsProcessingParameterNumber.Type.Integer if integer else QgsProcessingParameterNumber.Type.Double,
            defaultValue=default, minValue=minimum, optional=name in DEPENDENCIES, **kwargs,
        )
        parameter.setMetadata({"widget_wrapper": {"decimals": 9 if name == "SAM_LR" else 6}})
        definitions[name] = (parameter, group)
    for name in ORDER:
        parameter, group = definitions[name]
        parameter.setMetadata({**parameter.metadata(), "samstars_group": group})
        if name in HELP:
            parameter.setHelp(HELP[name])
        algorithm.add_advanced(parameter)


def training_payload(algorithm, parameters, context):
    values, options = {}, {}
    for name, _, choices, _ in ENUMS:
        value = algorithm.parameterAsEnum(parameters, name, context)
        if not 0 <= value < len(choices):
            raise QgsProcessingException(f"Choose a valid value for {name}.")
        values[name] = value
    for name, key, _, _, _ in BOOLEANS:
        values[name] = algorithm.parameterAsBool(parameters, name, context)
        if key:
            options[key] = values[name]
    for name, key, _, _, _, integer, _ in NUMBERS:
        if name in DEPENDENCIES:
            controller, expected = DEPENDENCIES[name]
            if values[controller] != expected:
                continue
        method = algorithm.parameterAsInt if integer else algorithm.parameterAsDouble
        options[key] = method(parameters, name, context)
    options["stardist_input_key"] = ("index", "file_name")[values["STARDIST_INPUT"]]
    if values["FAST_PROPOSALS"]:
        options["fast"] = values["FAST_PROPOSALS"] == 1
    try:
        return training_options(options)
    except ValueError as exc:
        raise QgsProcessingException(str(exc)) from exc
