"""Segment a single scene with a saved SAMStars bundle."""

from pathlib import Path

from qgis.core import (
    Qgis, QgsProcessingContext, QgsProcessingException,
    QgsProcessingOutputVectorLayer, QgsProcessingParameterFile,
    QgsProcessingUtils,
)

from .common import SamStarsAlgorithm


class SegmentAlgorithm(SamStarsAlgorithm):
    def name(self):
        return "segment"

    def displayName(self):
        return "Segment tree crowns"

    def shortHelpString(self):
        return (
            "Apply a trained SAMStars bundle to imagery, canopy heights, and first-return density. "
            "Use local rasters with overlapping coverage, north-up square imagery pixels, and a "
            "projected imagery CRS in metres. The CHM and density are aligned automatically. "
            "Band settings are read from the model unless overridden in Advanced parameters. "
            "Choose a new or empty output folder. The resulting crowns/crowns.gpkg is loaded "
            "into the project. CPU is the default; start with a small area. "
            "Set up the engine using Plugins → SAMStars → Set up SAMStars."
        )

    def initAlgorithm(self, config=None):
        self.add_scene_parameters()
        self.addParameter(QgsProcessingParameterFile(
            "MODEL", "Model bundle folder", behavior=QgsProcessingParameterFile.Behavior.Folder,
        ))
        self.add_output_folder("Segmentation output folder")
        self.addOutput(QgsProcessingOutputVectorLayer(
            "CROWNS", "Tree crowns", type=Qgis.ProcessingSourceType.VectorPolygon,
        ))

    def processAlgorithm(self, parameters, context, feedback):
        payload = self.scene_payload(parameters, context)
        model = Path(self.parameterAsFile(parameters, "MODEL", context)).expanduser().resolve()
        if not model.is_dir() or not (model / "manifest.json").is_file():
            raise QgsProcessingException("Choose a trained SAMStars bundle folder containing manifest.json.")
        payload["model"] = str(model)
        payload["out"] = self.output_path(
            parameters, context, [payload[key] for key in ("img", "chm", "den", "model")],
        )
        result = self.run_engine("segment", payload, feedback)
        if context.project() is not None and not feedback.isCanceled():
            details = QgsProcessingContext.LayerDetails(
                "Tree crowns", context.project(), "CROWNS", QgsProcessingUtils.LayerHint.Vector,
            )
            context.addLayerToLoadOnCompletion(result["CROWNS"], details)
        return result
