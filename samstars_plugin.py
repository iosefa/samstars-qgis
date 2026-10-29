"""Provider lifecycle and the single setup menu action."""

from qgis.core import Qgis, QgsApplication

from .provider import SamStarsProvider


class SamStarsPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.setup_action = None

    def initProcessing(self):
        if Qgis.QGIS_VERSION_INT < 40000:
            raise RuntimeError("SAMStars requires QGIS 4.0 or later.")
        if self.provider is None:
            provider = SamStarsProvider()
            if QgsApplication.processingRegistry().addProvider(provider):
                self.provider = provider

    def initGui(self):
        self.initProcessing()
        if self.iface is not None and self.setup_action is None:
            from qgis.PyQt.QtGui import QAction

            self.setup_action = QAction("Set up SAMStars…", self.iface.mainWindow())
            self.setup_action.triggered.connect(self.show_setup)
            self.iface.addPluginToMenu("&SAMStars", self.setup_action)

    def show_setup(self):
        from .setup_dialog import SetupDialog

        dialog = SetupDialog(self.iface.mainWindow())
        dialog.exec()

    def unload(self):
        if self.setup_action is not None:
            self.iface.removePluginMenu("&SAMStars", self.setup_action)
            self.setup_action.deleteLater()
            self.setup_action = None
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
