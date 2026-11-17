"""Individual SAMStars steps as ordinary Processing dialogs."""

from pathlib import Path

from qgis.core import (
    Qgis, QgsProcessingContext, QgsProcessingException, QgsProcessingOutputFile,
    QgsProcessingOutputRasterLayer, QgsProcessingOutputVectorLayer,
    QgsProcessingParameterBoolean, QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource, QgsProcessingParameterFile,
    QgsProcessingParameterNumber, QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString, QgsProcessingUtils, QgsVectorFileWriter,
)

from .common import SamStarsAlgorithm


class WorkflowStep(SamStarsAlgorithm):
    def group(self):
        return "Workflow steps"

    def groupId(self):
        return "steps"

    def raster(self, name, label):
        self.addParameter(QgsProcessingParameterRasterLayer(name, label))

    def number(self, name, label, default, *, minimum=0, maximum=None, integer=False, advanced=False):
        kwargs = {"defaultValue": default, "minValue": minimum}
        if maximum is not None:
            kwargs["maxValue"] = maximum
        parameter = QgsProcessingParameterNumber(
            name, label, type=QgsProcessingParameterNumber.Type.Integer if integer else
            QgsProcessingParameterNumber.Type.Double, **kwargs,
        )
        (self.add_advanced if advanced else self.addParameter)(parameter)

    def bands(self, *, model=False):
        self.add_advanced(QgsProcessingParameterString(
            "RGB_BANDS", "RGB bands (blank uses model settings)" if model else "RGB bands (red, green, blue)",
            defaultValue="" if model else "1,2,3", optional=model,
        ))
        if not model:
            self.number("NIR_BAND", "Near-infrared band (0 = unused)", 0, integer=True, advanced=True)
            self.number("TEXTURE_BAND", "Texture band (0 = mean RGB)", 0, integer=True, advanced=True)

    def model(self):
        self.addParameter(QgsProcessingParameterFile(
            "MODEL", "SAMStars model bundle folder", behavior=QgsProcessingParameterFile.Behavior.Folder,
        ))

    def raster_path(self, parameters, name, context, *, image=False):
        layer = self.parameterAsRasterLayer(parameters, name, context)
        if layer is None or not layer.isValid():
            raise QgsProcessingException(f"Choose a valid {name.lower()} raster.")
        path = Path(layer.source()).expanduser()
        if layer.providerType() != "gdal" or not path.is_file():
            raise QgsProcessingException(f"{name}: export the raster to a local GeoTIFF first.")
        if not layer.crs().isValid() or (not image and layer.bandCount() != 1):
            raise QgsProcessingException(f"{name} needs a defined CRS" + ("." if image else " and one band."))
        if image and (layer.bandCount() < 3 or layer.crs().isGeographic() or layer.crs().mapUnits() != Qgis.DistanceUnit.Meters):
            raise QgsProcessingException("Imagery needs at least three bands and a projected CRS in metres.")
        return str(path.resolve())

    def band_payload(self, parameters, context, *, model=False):
        result = {}
        rgb = self.parameterAsString(parameters, "RGB_BANDS", context).strip()
        if rgb:
            try:
                values = [int(value.strip()) for value in rgb.split(",")]
            except ValueError:
                values = []
            if len(values) != 3 or len(set(values)) != 3 or min(values) < 1:
                raise QgsProcessingException("Enter three distinct RGB band numbers, e.g. 1,2,3.")
            result["rgb_bands"] = values
        if not model:
            for name, key in (("NIR_BAND", "nir_band"), ("TEXTURE_BAND", "texture_band")):
                result[key] = self.parameterAsInt(parameters, name, context) or None
        return result

    def model_path(self, parameters, context):
        path = Path(self.parameterAsFile(parameters, "MODEL", context)).expanduser().resolve()
        if not path.is_dir() or not (path / "manifest.json").is_file():
            raise QgsProcessingException("Choose a trained SAMStars bundle containing manifest.json.")
        return str(path)

    def execute(self, payload, parameters, context, feedback, *, inputs, layers=()):
        payload["out"] = self.output_path(parameters, context, inputs)
        result = self.run_engine(self.name(), payload, feedback)
        if context.project() is not None and not feedback.isCanceled():
            for key, label, vector in layers:
                context.addLayerToLoadOnCompletion(result[key], QgsProcessingContext.LayerDetails(
                    label, context.project(), key,
                    QgsProcessingUtils.LayerHint.Vector if vector else QgsProcessingUtils.LayerHint.Raster,
                ))
        return result


class AlignRastersAlgorithm(WorkflowStep):
    def name(self):
        return "alignrasters"

    def displayName(self):
        return "Align input rasters"

    def shortHelpString(self):
        return (
            "Align CHM and first-return density to the imagery's CRS, extent, resolution, and pixel grid. "
            "Imagery must already be north-up with square pixels in a projected CRS in metres. "
            "Bilinear resampling matches SAMStars preparation. Density units must stay consistent with training. "
            "The original imagery is returned as a reference; its bands are preserved. This does not create "
            "CHM/density from point clouds or correct survey misregistration. Choose a new or empty output folder."
        )

    def initAlgorithm(self, config=None):
        for name, label in (("IMAGE", "Reference imagery"), ("CHM", "Canopy height (metres)"), ("DENSITY", "First-return density")):
            self.raster(name, label)
        self.add_output_folder("Aligned raster folder")
        for key, label in (("REFERENCE_IMAGE", "Reference imagery"), ("ALIGNED_CHM", "Aligned canopy height"), ("ALIGNED_DENSITY", "Aligned first-return density")):
            self.addOutput(QgsProcessingOutputRasterLayer(key, label))

    def processAlgorithm(self, parameters, context, feedback):
        payload = {key: self.raster_path(parameters, name, context, image=name == "IMAGE") for name, key in
                   (("IMAGE", "img"), ("CHM", "chm"), ("DENSITY", "den"))}
        return self.execute(payload, parameters, context, feedback, inputs=list(payload.values()), layers=(
            ("ALIGNED_CHM", "Aligned canopy height", False), ("ALIGNED_DENSITY", "Aligned density", False),
        ))


class CostSurfaceAlgorithm(WorkflowStep):
    def name(self):
        return "costsurface"

    def displayName(self):
        return "Build cost surface"

    def shortHelpString(self):
        return (
            "Build a 0–1 boundary-cost raster from aligned imagery and CHM. Higher costs indicate likely "
            "crown boundaries. Relative weights are normalized; the NDVI term is used only when a NIR band "
            "is selected under Advanced. RGB and texture band choices should match training. Run Align input "
            "rasters first if the grids differ. Output: cost.tif in a new or empty folder."
        )

    def initAlgorithm(self, config=None):
        self.raster("IMAGE", "Imagery")
        self.raster("CHM", "Aligned canopy height (metres)")
        self.bands()
        for key, label, value in (("HEIGHT_WEIGHT", "Height-edge weight", .4), ("NDVI_WEIGHT", "NDVI-gap weight", .3), ("TEXTURE_WEIGHT", "Texture weight", .3)):
            self.number(key, label, value)
        self.add_output_folder("Cost surface folder")
        self.addOutput(QgsProcessingOutputRasterLayer("COST", "Cost surface"))

    def processAlgorithm(self, parameters, context, feedback):
        payload = self.band_payload(parameters, context)
        payload.update(img=self.raster_path(parameters, "IMAGE", context, image=True),
                       chm=self.raster_path(parameters, "CHM", context))
        payload["weights"] = [self.parameterAsDouble(parameters, key, context) for key in ("HEIGHT_WEIGHT", "NDVI_WEIGHT", "TEXTURE_WEIGHT")]
        return self.execute(payload, parameters, context, feedback, inputs=[payload["img"], payload["chm"]],
                            layers=(("COST", "Cost surface", False),))


class SeedsAlgorithm(WorkflowStep):
    def name(self):
        return "generateseeds"

    def displayName(self):
        return "Generate seed points"

    def shortHelpString(self):
        return (
            "Detect peaks in aligned CHM and density rasters, then group candidates using the cost surface. "
            "The peak threshold applies to both height and density values, so density units matter. "
            "Outputs editable seed points. Inspect, move, add, or delete points before Generate crown proposals; "
            "that tool uses your chosen seed layer and does not regenerate seeds. "
            "All three rasters must share a projected metre grid. Choose a new or empty output folder."
        )

    def initAlgorithm(self, config=None):
        for name, label in (("CHM", "Aligned canopy height (metres)"), ("DENSITY", "Aligned first-return density"), ("COST", "Cost surface")):
            self.raster(name, label)
        self.number("PEAK_THRESHOLD", "Minimum peak value (CHM and density)", 4.5)
        self.number("MIN_CANOPY_HEIGHT", "Minimum canopy height (metres)", 2)
        self.number("SEARCH_RADIUS", "Peak-search radius (pixels)", 4, minimum=1, integer=True)
        self.number("SMOOTHING", "Gaussian smoothing (pixels)", 2, integer=True)
        self.number("MERGE_RADIUS", "Candidate merge radius (metres)", 3, minimum=.001)
        self.number("COST_WEIGHT", "Cost penalty for merging", 3, advanced=True)
        self.number("HEIGHT_SPLIT", "Height difference for cluster splitting (0 = disabled)", 1, advanced=True)
        self.number("THIN_DISTANCE", "Final seed thinning distance (metres, 0 = disabled)", .8, advanced=True)
        self.add_output_folder("Seed output folder")
        self.addOutput(QgsProcessingOutputVectorLayer("SEEDS", "Seed points", type=Qgis.ProcessingSourceType.VectorPoint))

    def processAlgorithm(self, parameters, context, feedback):
        payload = {key: self.raster_path(parameters, name, context) for name, key in (("CHM", "chm"), ("DENSITY", "den"), ("COST", "cost"))}
        inputs = list(payload.values())
        payload["seed_settings"] = {key: self.parameterAsDouble(parameters, name, context) for name, key in (
            ("PEAK_THRESHOLD", "d_min"), ("MIN_CANOPY_HEIGHT", "min_canopy_height"),
            ("MERGE_RADIUS", "merge_radius"), ("COST_WEIGHT", "cost_weight"), ("HEIGHT_SPLIT", "dz_merge"),
        )}
        for name, key in (("SEARCH_RADIUS", "min_dist_px"), ("SMOOTHING", "gauss_sigma")):
            payload["seed_settings"][key] = self.parameterAsInt(parameters, name, context)
        payload["thin_distance"] = self.parameterAsDouble(parameters, "THIN_DISTANCE", context)
        return self.execute(payload, parameters, context, feedback, inputs=inputs, layers=(("SEEDS", "Seed points", True),))


class ProposalsAlgorithm(WorkflowStep):
    def name(self):
        return "proposals"

    def displayName(self):
        return "Generate crown proposals (SAM)"

    def shortHelpString(self):
        return (
            "Generate SAM proposal rasters from aligned imagery, CHM, cost, and your seed points. "
            "Selections, layer filters, and unsaved seed edits are included. No cost surface or seeds are "
            "regenerated. RGB defaults come from the trained bundle. CPU is the default; select CUDA or MPS "
            "under Advanced after GPU setup. Pass Proposal records to Refine crowns. "
            "Outputs include probability and canopy-index rasters, not final crown polygons. "
            "Choose a new or empty output folder and keep it with its proposal records."
        )

    def initAlgorithm(self, config=None):
        for name, label in (("IMAGE", "Imagery"), ("CHM", "Aligned canopy height (metres)"), ("COST", "Cost surface")):
            self.raster(name, label)
        self.addParameter(QgsProcessingParameterFeatureSource("SEEDS", "Seed points", types=[Qgis.ProcessingSourceType.VectorPoint]))
        self.model()
        self.bands(model=True)
        self.add_advanced(QgsProcessingParameterEnum("DEVICE", "SAM compute device", options=["CPU", "CUDA", "Apple MPS"], defaultValue=0))
        self.add_advanced(QgsProcessingParameterBoolean("FAST", "Batch prompts across the scene (uses more GPU memory)", defaultValue=False))
        self.number("CROP_SMOOTHING", "Per-crop smoothing (pixels, -1 = model)", -1, minimum=-1, advanced=True)
        self.number("MOSAIC_SMOOTHING", "Proposal mosaic smoothing (pixels, -1 = model)", -1, minimum=-1, advanced=True)
        self.add_output_folder("Proposal output folder")
        self.addOutput(QgsProcessingOutputFile("PROPOSAL_RECORDS", "Proposal records"))
        self.addOutput(QgsProcessingOutputRasterLayer("PROBABILITY", "SAM probability"))
        self.addOutput(QgsProcessingOutputRasterLayer("CANOPY_INDEX", "SAM canopy index"))

    def export_seeds(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, "SEEDS", context)
        if source is None or not source.sourceCrs().isValid():
            raise QgsProcessingException("Choose seed points with a defined CRS.")
        path = QgsProcessingUtils.generateTempFilename("seeds.gpkg", context)
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName, options.layerName = "GPKG", "seeds"
        writer = QgsVectorFileWriter.create(path, source.fields(), source.wkbType(), source.sourceCrs(), context.transformContext(), options)
        count = 0
        try:
            if writer.hasError() != QgsVectorFileWriter.WriterError.NoError:
                raise QgsProcessingException(writer.errorMessage())
            for feature in source.getFeatures():
                if feedback.isCanceled():
                    raise QgsProcessingException("SAMStars processing canceled.")
                geometry = feature.geometry()
                if geometry.isNull() or geometry.isEmpty() or not geometry.isGeosValid():
                    raise QgsProcessingException("Seed points include empty or invalid geometry.")
                if not writer.addFeature(feature):
                    raise QgsProcessingException(writer.lastError())
                count += 1
        finally:
            del writer
        if not count:
            raise QgsProcessingException("Choose at least one seed point.")
        return path

    def processAlgorithm(self, parameters, context, feedback):
        payload = self.band_payload(parameters, context, model=True)
        for name, key in (("IMAGE", "img"), ("CHM", "chm"), ("COST", "cost")):
            payload[key] = self.raster_path(parameters, name, context, image=name == "IMAGE")
        payload["model"] = self.model_path(parameters, context)
        payload["seeds"] = self.export_seeds(parameters, context, feedback)
        inputs = [payload[key] for key in ("img", "chm", "cost", "model", "seeds")]
        device = self.parameterAsEnum(parameters, "DEVICE", context)
        if device not in (0, 1, 2):
            raise QgsProcessingException("Choose CPU, CUDA, or Apple MPS.")
        payload["device"] = ("cpu", "cuda", "mps")[device]
        payload["fast"] = self.parameterAsBool(parameters, "FAST", context)
        for name, key in (("CROP_SMOOTHING", "crop_smooth_sigma"), ("MOSAIC_SMOOTHING", "smooth_sigma")):
            value = self.parameterAsDouble(parameters, name, context)
            if value != -1:
                payload[key] = value
        return self.execute(payload, parameters, context, feedback, inputs=inputs, layers=(
            ("PROBABILITY", "SAM probability", False), ("CANOPY_INDEX", "SAM canopy index", False),
        ))


class RefineAlgorithm(WorkflowStep):
    def name(self):
        return "refine"

    def displayName(self):
        return "Refine crowns (StarDist)"

    def shortHelpString(self):
        return (
            "Refine saved proposal rasters into crown polygons without rerunning SAM. Select proposal_records.json "
            "from Generate crown proposals and the trained model bundle. Keep the proposal rasters available. "
            "Input-channel and threshold defaults come from the bundle. Set both thresholds together to override "
            "them. This step uses CPU. A new output folder lets you compare refinement settings while retaining "
            "the same proposals. Final crowns are loaded into QGIS."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFile("PROPOSAL_RECORDS", "Proposal records", fileFilter="JSON files (*.json)"))
        self.model()
        self.number("PROB_THRESHOLD", "Probability threshold (-1 = model)", -1, minimum=-1, maximum=1)
        self.number("NMS_THRESHOLD", "Overlap suppression threshold (-1 = model)", -1, minimum=-1, maximum=1)
        self.number("WINDOW_SIZE", "Refinement window size (metres, -1 = model, 0 = whole raster)", -1, minimum=-1, advanced=True)
        self.number("WINDOW_OVERLAP", "Window overlap (metres, -1 = model)", -1, minimum=-1, advanced=True)
        self.number("MIN_AREA", "Minimum crown area (square metres)", 0, advanced=True)
        self.number("EDGE_BUFFER", "Exclude crowns near raster edges (pixels)", .5, advanced=True)
        self.number("DEDUP_THRESHOLD", "Duplicate crown overlap threshold (-1 = model, 0 = disabled)", -1, minimum=-1, maximum=1, advanced=True)
        self.add_output_folder("Refinement output folder")
        self.addOutput(QgsProcessingOutputVectorLayer("CROWNS", "Tree crowns", type=Qgis.ProcessingSourceType.VectorPolygon))

    def processAlgorithm(self, parameters, context, feedback):
        records = Path(self.parameterAsFile(parameters, "PROPOSAL_RECORDS", context)).expanduser().resolve()
        if not records.is_file():
            raise QgsProcessingException("Choose an existing proposal_records.json file.")
        payload = {"records": str(records), "model": self.model_path(parameters, context)}
        for name, key in (
            ("PROB_THRESHOLD", "stardist_prob_thr"), ("NMS_THRESHOLD", "stardist_nms_thr"),
            ("WINDOW_SIZE", "stardist_window_size_m"), ("WINDOW_OVERLAP", "stardist_window_overlap_m"),
            ("DEDUP_THRESHOLD", "stardist_dedup_iou_thr"),
        ):
            value = self.parameterAsDouble(parameters, name, context)
            if value != -1:
                payload[key] = value
        if ("stardist_prob_thr" in payload) != ("stardist_nms_thr" in payload):
            raise QgsProcessingException("Set both refinement thresholds, or leave both at -1 to use the model.")
        payload["min_area"] = self.parameterAsDouble(parameters, "MIN_AREA", context)
        payload["edge_buffer"] = self.parameterAsDouble(parameters, "EDGE_BUFFER", context)
        return self.execute(payload, parameters, context, feedback, inputs=[payload["records"], payload["model"]],
                            layers=(("CROWNS", "Tree crowns", True),))
