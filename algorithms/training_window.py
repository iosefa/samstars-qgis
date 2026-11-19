"""Create a measured polygon guide for manually labeling training crowns."""

import math

from qgis.PyQt.QtCore import QMetaType, QThread
from qgis.core import (
    Qgis, QgsApplication, QgsCoordinateTransform, QgsCsException, QgsFeature,
    QgsFeatureSink, QgsField, QgsFields, QgsFillSymbol, QgsGeometry, QgsPointXY,
    QgsProcessingException, QgsProcessingUtils,
    QgsProcessingParameterCrs, QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink, QgsProcessingParameterNumber,
    QgsRectangle, QgsSingleSymbolRenderer,
)

from .common import SamStarsAlgorithm


class TrainingWindowAlgorithm(SamStarsAlgorithm):
    def name(self):
        return "createtrainingwindow"

    def displayName(self):
        return "Create training window"

    def group(self):
        return "Training preparation"

    def groupId(self):
        return "trainingpreparation"

    def flags(self):
        # Reading the map canvas must stay on the GUI thread.
        return super().flags() | Qgis.ProcessingAlgorithmFlag.NoThreading

    def shortHelpString(self):
        return (
            "Create a square polygon guide for drawing training crowns. Enter the side "
            "length in metres and choose a projected CRS in metres, preferably your "
            "imagery CRS. Use the current map-view centre, or enter X and Y in the "
            "chosen window CRS. Map-view coordinates are transformed automatically.\n\n"
            "The output has an outline with no fill. Save it to a GeoPackage to keep it. "
            "Create a separate polygon layer and draw one polygon per crown within "
            "the guide. Fully label the area, then crop the training rasters to it "
            "before running Train a model; unlabeled pixels are treated as background.\n\n"
            "This tool only creates the guide: it does not crop rasters, snap to their "
            "pixel grid, or set the model's SAM processing window. No engine is needed."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterNumber(
            "SIZE", "Window size (metres per side)",
            type=QgsProcessingParameterNumber.Type.Double, defaultValue=256, minValue=0,
        ))
        self.addParameter(QgsProcessingParameterCrs(
            "CRS", "Window CRS (projected, metres)", defaultValue="ProjectCrs",
        ))
        self.addParameter(QgsProcessingParameterEnum(
            "CENTER_SOURCE", "Window centre", options=[
                "Centre of current map view", "Enter coordinates in window CRS",
            ], defaultValue=0,
        ))
        for name, label in (("CENTER_X", "Centre X"), ("CENTER_Y", "Centre Y")):
            self.addParameter(QgsProcessingParameterNumber(
                name, label + " (entered coordinates only)",
                type=QgsProcessingParameterNumber.Type.Double, optional=True,
            ))
        self.addParameter(QgsProcessingParameterFeatureSink(
            "OUTPUT", "Training window", type=Qgis.ProcessingSourceType.VectorPolygon,
        ))

    def processAlgorithm(self, parameters, context, feedback):
        size = self.parameterAsDouble(parameters, "SIZE", context)
        if not math.isfinite(size) or size <= 0:
            raise QgsProcessingException("Window size must be a positive, finite number of metres.")
        crs = self.parameterAsCrs(parameters, "CRS", context)
        if not crs.isValid() or crs.type() != Qgis.CrsType.Projected or crs.mapUnits() != Qgis.DistanceUnit.Meters:
            raise QgsProcessingException("Choose a projected window CRS with metre units, such as the imagery CRS.")

        source = self.parameterAsEnum(parameters, "CENTER_SOURCE", context)
        if source == 0:
            if QThread.currentThread() != QgsApplication.instance().thread():
                raise QgsProcessingException("Map-view centre requires the QGIS main thread. Use entered coordinates instead.")
            from qgis import utils
            canvas = utils.iface.mapCanvas() if utils.iface is not None else None
            if canvas is None or canvas.extent().isEmpty():
                raise QgsProcessingException("Open a QGIS map view, or choose entered coordinates for the window centre.")
            map_crs = canvas.mapSettings().destinationCrs()
            if not map_crs.isValid():
                raise QgsProcessingException("The map view needs a defined CRS, or use entered coordinates.")
            try:
                center = QgsCoordinateTransform(map_crs, crs, context.transformContext()).transform(canvas.extent().center())
            except QgsCsException as exc:
                raise QgsProcessingException(f"Could not transform the map-view centre to the window CRS: {exc}") from exc
        elif source == 1:
            coordinates = []
            for name in ("CENTER_X", "CENTER_Y"):
                try:
                    value = float(self.parameterAsString(parameters, name, context))
                except (TypeError, ValueError) as exc:
                    raise QgsProcessingException("Enter both centre X and centre Y in the window CRS.") from exc
                if not math.isfinite(value):
                    raise QgsProcessingException("Centre coordinates must be finite numbers.")
                coordinates.append(value)
            center = QgsPointXY(*coordinates)
        else:
            raise QgsProcessingException("Choose the current map view or entered coordinates for the window centre.")

        bounds = (center.x() - size / 2, center.y() - size / 2,
                  center.x() + size / 2, center.y() + size / 2)
        if not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise QgsProcessingException("The window centre and size must produce finite, distinct corner coordinates.")
        if feedback.isCanceled():
            return {}
        fields = QgsFields()
        fields.append(QgsField("window_id", QMetaType.Type.Int))
        for name in ("size_m", "center_x", "center_y"):
            fields.append(QgsField(name, QMetaType.Type.Double))
        sink, destination = self.parameterAsSink(
            parameters, "OUTPUT", context, fields, Qgis.WkbType.Polygon, crs,
        )
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, "OUTPUT"))
        feature = QgsFeature(fields)
        feature.setAttributes([1, size, center.x(), center.y()])
        feature.setGeometry(QgsGeometry.fromRect(QgsRectangle(*bounds)))
        if not sink.addFeature(feature, QgsFeatureSink.Flag.FastInsert):
            raise QgsProcessingException(f"Could not write the training window: {sink.lastError()}")
        sink.finalize()
        self._destination = destination
        feedback.pushInfo(
            f"Created a {size:g} m square centred at {center.x():g}, {center.y():g}. "
            "Draw crown polygons in a separate layer and crop your rasters to the fully labeled area before training."
        )
        return {"OUTPUT": destination}

    def postProcessAlgorithm(self, context, feedback):
        destination = getattr(self, "_destination", None)
        if destination and context.willLoadLayerOnCompletion(destination):
            layer = QgsProcessingUtils.mapLayerFromString(destination, context)
            if layer is not None:
                layer.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({
                    "style": "no", "outline_color": "255,128,0,255",
                    "outline_width": "0.6", "outline_width_unit": "MM",
                    "joinstyle": "miter",
                })))
        return {}
