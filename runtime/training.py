"""Validate training options without importing QGIS or scientific libraries."""

import math

LABELS = {
    "chip_size": "SAM training window size", "chip_stride": "SAM window stride",
    "sam_window_size_m": "SAM window size in metres", "sam_lr": "SAM learning rate",
    "sam_epochs": "SAM epochs", "stardist_epochs": "StarDist epochs",
    "fg_extra": "Extra foreground seeds",
    "stardist_patch_size": "StarDist patch size", "val_fraction": "Validation fraction",
    "stardist_window_size_m": "Refinement window size", "stardist_window_overlap_m": "Refinement overlap",
    "stardist_dedup_iou_thr": "Duplicate crown IoU threshold",
}


def training_options(values, version=None):
    """Normalize active API options; preserve omitted engine defaults."""
    options = dict(values)

    def number(key, minimum=0, *, integer=False, positive=False, below=None):
        if key not in options:
            return None
        value = options[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (integer and not isinstance(value, int))
                or value < minimum or (positive and value <= 0)
                or (below is not None and value >= below)):
            if integer:
                rule = "a positive integer" if minimum == 1 else "a nonnegative integer"
            else:
                rule = "a positive finite number" if positive else "a valid finite number"
            raise ValueError(f"{LABELS.get(key, key.replace('_', ' '))} must be {rule} (minimum {minimum}" +
                             (f", less than {below}" if below is not None else "") + ").")
        return value

    for key in ("thin_seeds", "stardist_use_aux", "stardist_normalize_inputs", "stardist_augment", "fast"):
        if key in options and not isinstance(options[key], bool):
            raise ValueError(f"{key} must be enabled or disabled.")
    if options.get("thin_seeds") is False:
        options.pop("thin_dist", None)
    if options.get("sam_window_size_m") is not None:
        if options.get("chip_size") is not None:
            raise ValueError("Choose a SAM window in pixels or metres, not both.")
        options.pop("chip_size", None)
        number("sam_window_size_m", positive=True)
        size = None
    else:
        options.pop("sam_window_size_m", None)
        options.setdefault("chip_size", 512)
        size = number("chip_size", 1, integer=True)
    if options.get("chip_stride") is None:
        options.pop("chip_stride", None)
        if size is not None:
            options["chip_stride"] = max(1, size * 3 // 4)
    stride = number("chip_stride", 1, integer=True)
    if size is not None and stride is not None and stride > size:
        raise ValueError("SAM window stride must not exceed the window size; it would leave gaps.")

    for key in ("sam_epochs", "stardist_epochs", "seed_min_dist_px", "seed_min_samples", "stardist_patch_size"):
        number(key, 1, integer=True)
    for key in ("sam_min_iters", "stardist_min_iters", "seed_gauss_sigma", "fg_extra"):
        number(key, integer=True)
    for key in ("crop_smooth_sigma", "smooth_sigma", "post_smooth_sigma", "thin_dist", "seed_d_min",
                "seed_min_canopy_height"):
        number(key)
    for key in ("sam_lr", "seed_merge_radius"):
        number(key, positive=True)
    number("val_fraction", below=1)
    if "stardist_patch_size" in options and options["stardist_patch_size"] % 16:
        raise ValueError("StarDist training patch size must be a positive multiple of 16 pixels.")
    if options.get("stardist_input_key", "index") not in ("index", "file_name"):
        raise ValueError("Choose canopy index or SAM probability as the StarDist input.")
    if options.get("stardist_window_size_m") in (None, 0):
        options.pop("stardist_window_size_m", None)
        options.pop("stardist_window_overlap_m", None)
    else:
        window = number("stardist_window_size_m", positive=True)
        overlap = number("stardist_window_overlap_m") or 0
        if overlap >= window:
            raise ValueError("Refinement overlap must be smaller than the refinement window size.")
    number("stardist_dedup_iou_thr", below=1)

    if version == "0.2.0":
        unsupported = [key for key in ("sam_window_size_m", "stardist_patch_size") if key in options]
        if "fg_extra" in options:
            if options["fg_extra"] == 80:
                # Stable has a fixed count of 80 and no fg_extra API argument.
                options.pop("fg_extra")
            else:
                unsupported.append(LABELS["fg_extra"])
        if size != 512:
            unsupported.append("SAM training window size")
        if unsupported:
            raise ValueError(
                "These training settings require SAMStars 0.3.0a1: " + ", ".join(unsupported) +
                ". Select Experimental 0.3.0a1 in SAMStars Setup and install or repair the engine, "
                "or use a 512-pixel SAM window, 80 extra foreground seeds, and disable the patch-size override with Stable 0.2.0."
            )
    return options
