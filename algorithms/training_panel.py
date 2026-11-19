"""Grouped native Processing widgets with conditional training controls."""

from qgis.PyQt.QtWidgets import QGroupBox, QVBoxLayout
from qgis.core import Qgis
from qgis.gui import QgsCollapsibleGroupBox
from processing.gui.ParametersPanel import ParametersPanel

try:
    from processing.gui.algorithm_widget import AlgorithmWidget
except ModuleNotFoundError as exc:
    if exc.name != "processing.gui.algorithm_widget":
        raise
    # QGIS 4.0 uses the same panel hook in its older dialog class.
    from processing.gui.AlgorithmDialog import AlgorithmDialog as AlgorithmWidget

from .training_parameters import DEPENDENCIES, GROUPS


class TrainingAlgorithmWidget(AlgorithmWidget):
    def getParametersPanel(self, algorithm, parent):
        return TrainingParametersPanel(parent, algorithm)


class TrainingParametersPanel(ParametersPanel):
    def __init__(self, parent, algorithm):
        super().__init__(parent, algorithm)
        # QGIS can expose its native collapsible box through the QGroupBox
        # binding, so a lookup by the Python subclass misses it on Qt 6.
        advanced = self.findChild(QGroupBox, "grpAdvanced")
        self.training_groups = {}
        if advanced is not None:
            for key, title in GROUPS.items():
                box = QgsCollapsibleGroupBox(title, advanced)
                box.setSaveCollapsedState(False)
                box.setCollapsed(key != "sam")
                layout = QVBoxLayout(box)
                advanced.layout().addWidget(box)
                self.training_groups[key] = box
                for parameter in algorithm.parameterDefinitions():
                    if not parameter.flags() & Qgis.ProcessingParameterFlag.Advanced:
                        continue
                    group = parameter.metadata().get("samstars_group", "inputs")
                    if group != key:
                        continue
                    wrapper = self.wrappers[parameter.name()]
                    for widget in (wrapper.wrappedLabel(), wrapper.wrappedWidget()):
                        if widget is not None:
                            layout.addWidget(widget)
        for name in {controller for controller, _ in DEPENDENCIES.values()}:
            self.wrappers[name].widgetValueHasChanged.connect(self.update_conditions)
        self.update_conditions()

    def update_conditions(self, *args):
        for name, (controller, expected) in DEPENDENCIES.items():
            enabled = self.wrappers[controller].parameterValue() == expected
            wrapper = self.wrappers[name]
            for widget in (wrapper.wrappedLabel(), wrapper.wrappedWidget()):
                if widget is not None:
                    widget.setEnabled(enabled)
