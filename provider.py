"""A small Processing provider; scientific imports stay in the engine."""

from pathlib import Path

from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider

from .algorithms.dependency_check import DependencyCheckAlgorithm
from .algorithms.segment import SegmentAlgorithm
from .algorithms.train import TrainAlgorithm
from .algorithms.training_window import TrainingWindowAlgorithm
from .algorithms.steps import AlignRastersAlgorithm, CostSurfaceAlgorithm, SeedsAlgorithm, ProposalsAlgorithm, RefineAlgorithm


class SamStarsProvider(QgsProcessingProvider):
    def id(self):
        return "samstars"

    def name(self):
        return "SAMStars"

    def longName(self):
        return self.name()

    def icon(self):
        return QIcon(str(Path(__file__).parent / "icons" / "icon.svg"))

    def loadAlgorithms(self):
        self.addAlgorithm(SegmentAlgorithm())
        self.addAlgorithm(TrainAlgorithm())
        self.addAlgorithm(TrainingWindowAlgorithm())
        self.addAlgorithm(DependencyCheckAlgorithm())
        for algorithm in (AlignRastersAlgorithm, CostSurfaceAlgorithm, SeedsAlgorithm, ProposalsAlgorithm, RefineAlgorithm):
            self.addAlgorithm(algorithm())
