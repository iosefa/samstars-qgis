"""Shared parameters and input validation for the two scene workflows."""

from pathlib import Path

from qgis.core import (
    Qgis, QgsProcessingAlgorithm, QgsProcessingException,
    QgsProcessingParameterBoolean, QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination, QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer, QgsProcessingParameterString,
)

from ..runtime.client import EngineError, call_engine
from ..runtime.worker import validate_output


class SamStarsAlgorithm(QgsProcessingAlgorithm):
    def group(self):
        return "Tree crowns"

    def groupId(self):
        return "crowns"

    def createInstance(self):
        return type(self)()

    def helpUrl(self):
        return (Path(__file__).resolve().parents[1] / "docs" / "index.html").as_uri() + "#" + self.name()

    def add_advanced(self, parameter):
        parameter.setFlags(parameter.flags() | Qgis.ProcessingParameterFlag.Advanced)
        self.addParameter(parameter)

    def add_scene_parameters(self, *, training=False):
        self.addParameter(QgsProcessingParameterRasterLayer("IMAGE", "Imagery"))
        self.addParameter(QgsProcessingParameterRasterLayer("CHM", "Canopy height (metres)"))
        self.addParameter(QgsProcessingParameterRasterLayer("DENSITY", "First-return density"))
        self.add_advanced(QgsProcessingParameterString(
            "RGB_BANDS", "RGB bands (red, green, blue)" if training else "RGB bands (blank uses model settings)",
            defaultValue="1,2,3" if training else "", optional=not training,
        ))
        for name, label in (("NIR_BAND", "Near-infrared band"), ("TEXTURE_BAND", "Texture band")):
            self.add_advanced(QgsProcessingParameterNumber(
                name, label + (" (0 = unused)" if training else " (0 = model setting)"),
                type=QgsProcessingParameterNumber.Type.Integer, defaultValue=0, minValue=0,
            ))
        self.add_advanced(QgsProcessingParameterEnum(
            "DEVICE", "Compute device", options=["CPU", "CUDA", "Apple MPS"], defaultValue=0,
        ))
        self.add_advanced(QgsProcessingParameterBoolean(
            "KEEP_INTERMEDIATES", "Keep intermediate files", defaultValue=False,
        ))

    def add_output_folder(self, label):
        self.addParameter(QgsProcessingParameterFolderDestination("OUTPUT", label))

    def scene_payload(self, parameters, context):
        payload = {}
        layers = {}
        for name, key in (("IMAGE", "img"), ("CHM", "chm"), ("DENSITY", "den")):
            layer = self.parameterAsRasterLayer(parameters, name, context)
            if layer is None or not layer.isValid():
                raise QgsProcessingException(f"Select a valid {name.lower()} raster.")
            path = Path(layer.source()).expanduser()
            if layer.providerType() != "gdal" or not path.is_file():
                raise QgsProcessingException(
                    f"{name}: export the raster to a local GeoTIFF first. Web services and subdataset URIs are not supported."
                )
            payload[key] = str(path.resolve())
            layers[name] = layer
        image = layers["IMAGE"]
        if not image.crs().isValid() or image.crs().isGeographic() or image.crs().mapUnits() != Qgis.DistanceUnit.Meters:
            raise QgsProcessingException("Imagery needs a projected CRS in metres. Reproject the raster first.")
        if image.bandCount() < 3:
            raise QgsProcessingException("Imagery must contain at least three RGB bands.")
        for name in ("CHM", "DENSITY"):
            if layers[name].bandCount() != 1 or not layers[name].crs().isValid():
                raise QgsProcessingException(f"{name} must have one band and a defined CRS.")
        rgb = self.parameterAsString(parameters, "RGB_BANDS", context).strip()
        if rgb:
            try:
                bands = [int(value.strip()) for value in rgb.split(",")]
            except ValueError:
                bands = []
            if len(bands) != 3 or len(set(bands)) != 3 or any(not 1 <= band <= image.bandCount() for band in bands):
                raise QgsProcessingException("Enter three distinct image band numbers in red, green, blue order, e.g. 1,2,3.")
            payload["rgb_bands"] = bands
        for name, key in (("NIR_BAND", "nir_band"), ("TEXTURE_BAND", "texture_band")):
            band = self.parameterAsInt(parameters, name, context)
            if not 0 <= band <= image.bandCount():
                raise QgsProcessingException(f"{name} exceeds the imagery band count.")
            if band:
                payload[key] = band
        device = self.parameterAsEnum(parameters, "DEVICE", context)
        if device not in (0, 1, 2):
            raise QgsProcessingException("Choose CPU, CUDA, or Apple MPS.")
        payload["device"] = ("cpu", "cuda", "mps")[device]
        payload["storage"] = "persistent" if self.parameterAsBool(parameters, "KEEP_INTERMEDIATES", context) else "temp"
        return payload

    def output_path(self, parameters, context, inputs):
        path = self.parameterAsString(parameters, "OUTPUT", context)
        if not path or path == "TEMPORARY_OUTPUT":
            path = self.parameterDefinition("OUTPUT").generateTemporaryDestination(context)
        try:
            return str(validate_output(path, inputs))
        except (ValueError, OSError) as exc:
            raise QgsProcessingException(str(exc)) from exc

    def run_engine(self, operation, payload, feedback):
        if feedback.isCanceled():
            raise QgsProcessingException("SAMStars processing canceled.")
        try:
            return call_engine(operation, payload, feedback)
        except EngineError as exc:
            raise QgsProcessingException(str(exc)) from exc
