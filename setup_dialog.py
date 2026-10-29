"""Small asynchronous engine setup dialog; analysis stays in Processing."""

import os
from pathlib import Path
import threading

from qgis.PyQt.QtCore import QThread, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
)
from qgis.core import QgsApplication, QgsSettings

from .runtime import manager


class SetupThread(QThread):
    message = pyqtSignal(str)
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.cancel_event = threading.Event()

    def run(self):
        try:
            result = self.operation(
                cancel=self.cancel_event.is_set,
                event=lambda event: self.message.emit(str(event.get("message", ""))),
            )
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class SetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAMStars setup")
        self.resize(580, 400)
        self._thread = None
        self._closing = False
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Choose Python 3.10–3.12, then install a private SAMStars engine. "
            "If that environment already has SAMStars 0.2.0, choose Use existing. "
            "Installation needs internet access and several GB of disk space. "
            "All analysis runs through the Processing Toolbox."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        label = QLabel("Python executable")
        layout.addWidget(label)
        row = QHBoxLayout()
        self.python = QLineEdit(manager.engine_python() or manager.suggested_python())
        self.python.setPlaceholderText("/path/to/environment/bin/python")
        label.setBuddy(self.python)
        self.browse = QPushButton("Browse…")
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.python)
        row.addWidget(self.browse)
        layout.addLayout(row)
        self.status = QLabel(
            "Configured engine: " + (manager.engine_python() or "none")
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        if os.environ.get("SAMSTARS_PYTHON"):
            override = QLabel("SAMSTARS_PYTHON overrides the saved engine selection for this QGIS session.")
            override.setWordWrap(True)
            layout.addWidget(override)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        layout.addWidget(self.log)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        actions = QHBoxLayout()
        self.install = QPushButton("Install private engine")
        self.existing = QPushButton("Use existing")
        self.cancel = QPushButton("Cancel")
        self.help = QPushButton("Help")
        self.close_button = QPushButton("Close")
        for button in (self.install, self.existing, self.cancel, self.help, self.close_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.cancel.setEnabled(False)
        self.install.clicked.connect(self._install)
        self.existing.clicked.connect(self._existing)
        self.cancel.clicked.connect(self._cancel)
        self.help.clicked.connect(self._help)
        self.close_button.clicked.connect(self.close)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Python executable", self.python.text())
        if path:
            self.python.setText(path)

    def _help(self):
        url = QUrl.fromLocalFile(str(Path(__file__).parent / "docs" / "index.html"))
        url.setFragment("setup")
        QDesktopServices.openUrl(url)

    def _install(self):
        python = self.python.text().strip()
        root = Path(QgsApplication.qgisSettingsDirPath()) / "samstars" / "engines"
        self._start(lambda **kwargs: manager.install_engine(python, root, **kwargs))

    def _existing(self):
        python = self.python.text().strip()
        self._start(lambda **kwargs: manager.check_engine(python, **kwargs))

    def _start(self, operation):
        if self._thread is not None:
            return
        self.log.clear()
        self.status.setText("Checking the engine…")
        self.progress.setRange(0, 0)
        for widget in (self.python, self.browse, self.install, self.existing, self.close_button):
            widget.setEnabled(False)
        self.cancel.setEnabled(True)
        self._thread = SetupThread(operation, self)
        self._thread.message.connect(self.log.appendPlainText)
        self._thread.succeeded.connect(self._succeeded)
        self._thread.failed.connect(self._failed)
        self._thread.finished.connect(self._finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _succeeded(self, result):
        QgsSettings().setValue(manager.SETTING, result["python"])
        self.python.setText(result["python"])
        self.status.setText("Engine ready. Open Processing → Toolbox → SAMStars.")
        self.log.appendPlainText("Validated engine: " + result["python"])

    def _failed(self, message):
        self.status.setText("Setup did not complete. The previous selection is unchanged.")
        self.log.appendPlainText(message)

    def _cancel(self):
        if self._thread is not None:
            self._thread.cancel_event.set()
            self.cancel.setEnabled(False)
            self.status.setText("Canceling setup…")

    def _finished(self):
        self._thread = None
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        for widget in (self.python, self.browse, self.install, self.existing, self.close_button):
            widget.setEnabled(True)
        self.cancel.setEnabled(False)
        if self._closing:
            self.close()

    def reject(self):
        if self._thread is not None:
            self._closing = True
            self._cancel()
        else:
            super().reject()

    def closeEvent(self, event):
        if self._thread is not None:
            self._closing = True
            self._cancel()
            event.ignore()
        else:
            event.accept()
