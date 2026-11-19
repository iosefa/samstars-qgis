"""Train a reusable bundle from a labeled scene."""

from pathlib import Path

from qgis.core import (
    Qgis, QgsProcessingException, QgsProcessingOutputFolder,
    QgsProcessingParameterFeatureSource, QgsProcessingParameterFile,
    QgsVectorFileWriter,
    QgsProcessingUtils,
)

from .common import SamStarsAlgorithm
from .training_parameters import add_training_parameters, training_payload


class TrainAlgorithm(SamStarsAlgorithm):
    def name(self):
        return "train"

    def displayName(self):
        return "Train a model"

    def shortHelpString(self):
        return (
            "Train SAMStars from imagery, canopy height in metres, first-return density, and one "
            "polygon per labeled crown. Crop inputs to a completely annotated area. "
            "Supply the SAM ViT-L checkpoint (sam_vit_l_0b3195.pth). "
            "Training polygons are exported to a temporary GeoPackage, including any Processing "
            "feature selection. RGB bands default to 1,2,3; change Advanced parameters for "
            "another band order or optional NIR. Advanced groups expose SAM windows and learning, "
            "seed generation, proposal processing, StarDist training, and saved prediction defaults. "
            "SAM windows default to 512 pixels with 25% overlap. Custom sizes, metre-sized windows, "
            "patch-size overrides, and extra foreground seed counts other than 80 require SAMStars 0.3.0a1. "
            "Extra foreground seeds defaults to 80; set 0 for seed-only foreground prompts. "
            "Inactive fields are ignored. "
            "Validation splits records; a single record reuses training data for validation. "
            "Choose a new or empty model folder. "
            "The saved bundle can be used by Segment tree crowns. Training can take substantial "
            "time and memory; use the SAMStars Python API for larger tiled workflows."
        )

    def initAlgorithm(self, config=None):
        self.add_scene_parameters(training=True)
        self.addParameter(QgsProcessingParameterFeatureSource(
            "CROWNS", "Labeled tree crowns", types=[Qgis.ProcessingSourceType.VectorPolygon],
        ))
        self.addParameter(QgsProcessingParameterFile(
            "SAM_CHECKPOINT", "SAM ViT-L checkpoint", fileFilter="PyTorch checkpoint (*.pth *.pt)",
        ))
        add_training_parameters(self)
        self.add_output_folder("Model bundle folder")
        self.addOutput(QgsProcessingOutputFolder("MODEL", "Trained model bundle"))

    def createCustomParametersWidget(self, parent=None):
        from .training_panel import TrainingAlgorithmWidget

        return TrainingAlgorithmWidget(self, parent=parent)

    def processAlgorithm(self, parameters, context, feedback):
        payload = self.scene_payload(parameters, context)
        payload.update(training_payload(self, parameters, context))
        checkpoint = Path(self.parameterAsFile(parameters, "SAM_CHECKPOINT", context)).expanduser().resolve()
        if not checkpoint.is_file():
            raise QgsProcessingException("Choose the local SAM ViT-L checkpoint file.")
        payload["sam_base_ckpt"] = str(checkpoint)
        source = self.parameterAsSource(parameters, "CROWNS", context)
        if source is None or not source.sourceCrs().isValid():
            raise QgsProcessingException("Choose labeled crown polygons with a defined CRS.")
        # Always export the requested source: preserves subsets, selected features,
        # unsaved edits, and the intended layer of multi-layer GeoPackages.
        crowns = QgsProcessingUtils.generateTempFilename("training_crowns.gpkg", context)
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = "crowns"
        writer = QgsVectorFileWriter.create(
            crowns, source.fields(), source.wkbType(), source.sourceCrs(),
            context.transformContext(), options,
        )
        if writer.hasError() != QgsVectorFileWriter.WriterError.NoError:
            raise QgsProcessingException(writer.errorMessage())
        count = 0
        try:
            for feature in source.getFeatures():
                if feedback.isCanceled():
                    raise QgsProcessingException("SAMStars processing canceled.")
                geometry = feature.geometry()
                if geometry.isNull() or geometry.isEmpty() or not geometry.isGeosValid():
                    raise QgsProcessingException("Training crowns contain empty or invalid geometry. Fix geometries first.")
                if not writer.addFeature(feature):
                    raise QgsProcessingException(writer.lastError())
                count += 1
        finally:
            del writer
        if not count:
            raise QgsProcessingException("Training requires at least one labeled crown polygon.")
        payload["crowns"] = crowns
        payload["out"] = self.output_path(
            parameters, context, [payload[key] for key in ("img", "chm", "den", "crowns", "sam_base_ckpt")],
        )
        return self.run_engine("train", payload, feedback)
