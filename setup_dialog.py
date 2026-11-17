"""Compact engine setup, following the Coregix Processing plugin."""

from pathlib import Path
import threading

from qgis.PyQt.QtCore import QThread, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLayout, QLineEdit, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
)
from qgis.core import QgsSettings
from qgis.gui import QgsCollapsibleGroupBox

from .runtime import accelerators, manager
from .runtime.client import EngineCanceled


class SetupThread(QThread):
    progress = pyqtSignal(float, str)
    message = pyqtSignal(str)
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str, bool)

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.cancel_event = threading.Event()

    def _event(self, event):
        message = str(event.get("message", ""))
        if event.get("type") == "progress":
            self.progress.emit(float(event["value"]), message)
        elif message:
            self.message.emit(message)

    def run(self):
        try:
            result = self.operation(cancel=self.cancel_event.is_set, event=self._event)
            self.succeeded.emit(result)
        except EngineCanceled as exc:
            self.failed.emit(str(exc), True)
        except Exception as exc:
            self.failed.emit(str(exc), False)


class GpuSetupDialog(QDialog):
    """Choose an opt-in GPU profile or test the selected engine without changes."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAMStars GPU Setup")
        self.setMinimumWidth(500)
        self.state = state
        self.operation = ""
        layout = QVBoxLayout(self)
        for text in (
            "GPU setup is experimental. It accelerates SAM proposals and SAM training. "
            "StarDist refinement stays on CPU in private engines.",
            "Installation creates a new private engine and tests it before selection. "
            "It downloads several GB and can take several minutes; your previous engine is kept. NVIDIA drivers "
            "must already be installed. Setup does not change system drivers or QGIS.",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            layout.addWidget(label)
        self.device = QComboBox()
        self.device.addItem("CPU (default)", "cpu")
        self.device.addItem("NVIDIA CUDA — Linux x64", "cuda")
        self.device.addItem("Apple MPS — Apple Silicon macOS", "mps")
        label = QLabel("Engine type")
        label.setBuddy(self.device)
        layout.addWidget(label)
        layout.addWidget(self.device)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.install_button = QPushButton("Install GPU engine")
        self.test_button = QPushButton("Test selected engine")
        self.close_button = QPushButton("Cancel")
        actions = QHBoxLayout()
        for button in (self.install_button, self.test_button, self.close_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.install_button.clicked.connect(lambda: self._choose("install"))
        self.test_button.clicked.connect(lambda: self._choose("test"))
        self.close_button.clicked.connect(self.reject)
        self.device.currentIndexChanged.connect(self._refresh)
        default = state.get("accelerator", "cpu")
        if default == "cpu":
            default = "mps" if not accelerators.platform_error("mps") else "cuda"
        self.device.setCurrentIndex(self.device.findData(default))
        self._refresh()

    def _refresh(self):
        device = self.device.currentData()
        error = accelerators.platform_error(device)
        details = {
            "cpu": "Build a CPU engine. Processing tools continue to default to CPU.",
            "cuda": "Setup checks nvidia-smi and the driver, then chooses a CUDA runtime for the detected cards. "
                    "AMD and Intel GPUs are not supported by this profile.",
            "mps": "Requires macOS 14+ and native arm64 Python. Uses PyTorch MPS; TensorFlow Metal is not installed.",
        }[device]
        if self.state["override"]:
            details = "SAMSTARS_PYTHON controls the engine. You can test it, but cannot replace it here."
        elif error or self.state["error"]:
            details = error or self.state["error"]
        self.details.setText(details)
        self.install_button.setText("Install CPU engine" if device == "cpu" else "Install GPU engine")
        self.install_button.setEnabled(not error and not self.state["error"] and not self.state["override"])
        self.test_button.setEnabled(device != "cpu" and self.state["installed"])

    def _choose(self, operation):
        self.operation = operation
        self.accept()


class SetupDialog(QDialog):
    completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAMStars Setup")
        self.setMinimumWidth(560)
        self._thread = None
        self._closing = False
        self._removing = False
        self._diagnosing = False
        self._outcome = ""
        self._successful = False
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        explanation = QLabel(
            "SAMStars runs in a private Python engine. Install it once, then use "
            "the tools in the Processing Toolbox. Setup keeps the scientific "
            "libraries separate from QGIS."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.location = QLabel()
        self.location.setWordWrap(True)
        layout.addWidget(self.location)
        self.gpu_note = QLabel(
            "CPU is the default. GPU setup can accelerate SAM proposals; "
            "private engines keep StarDist refinement on CPU."
        )
        self.gpu_note.setWordWrap(True)
        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.gpu_note, 1)
        self.gpu_button = QPushButton("GPU setup…")
        self.gpu_button.clicked.connect(self._gpu_setup)
        gpu_row.addWidget(self.gpu_button)
        layout.addLayout(gpu_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        self.advanced = QgsCollapsibleGroupBox("Advanced")
        self.advanced.setSaveCollapsedState(False)
        advanced_layout = QVBoxLayout(self.advanced)
        label = QLabel("Python executable (optional for installation)")
        advanced_layout.addWidget(label)
        row = QHBoxLayout()
        self.python = QLineEdit()
        self.python.setPlaceholderText("Automatically find Python 3.10–3.12")
        label.setBuddy(self.python)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.python)
        row.addWidget(self.browse_button)
        advanced_layout.addLayout(row)
        hint = QLabel(
            "Installation needs Python 3.10–3.12 with venv support. To use an "
            "environment that already contains compatible SAMStars libraries, "
            "select its Python executable and click Use existing environment."
        )
        hint.setWordWrap(True)
        advanced_layout.addWidget(hint)
        self.existing_button = QPushButton("Use existing environment")
        self.existing_button.clicked.connect(self._existing)
        advanced_layout.addWidget(self.existing_button)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setFixedHeight(120)
        self.log.setPlaceholderText("Detailed setup log")
        advanced_layout.addWidget(self.log)
        self.advanced.setCollapsed(True)
        layout.addWidget(self.advanced)

        actions = QHBoxLayout()
        self.install_button = QPushButton("Install engine")
        self.remove_button = QPushButton("Remove engine")
        self.cancel_button = QPushButton("Cancel")
        self.help_button = QPushButton("Help")
        self.close_button = QPushButton("Close")
        for button in (
            self.install_button, self.remove_button, self.cancel_button,
            self.help_button, self.close_button,
        ):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.install_button.clicked.connect(self._install)
        self.remove_button.clicked.connect(self._remove)
        self.cancel_button.clicked.connect(self._cancel)
        self.help_button.clicked.connect(self._help)
        self.close_button.clicked.connect(self.close)
        self._refresh()

    def _refresh(self):
        state = manager.status()
        if state["installed"]:
            summary = "Private engine selected." if state["managed"] else "An existing Python environment is selected."
            if state["managed"] and state.get("accelerator") in ("cuda", "mps"):
                summary += " GPU profile: " + state["accelerator"].upper() + "; refinement: CPU."
        else:
            summary = "No working engine is selected."
        self.summary.setText(
            summary + f"\nPlatform: {state['platform']}. "
            "Installation needs internet access and several GB of free disk space."
        )
        self.location.setText(
            "Selected Python: " + state["python"]
            if state["installed"] else "Private engine location: " + state["path"]
        )
        self.install_button.setText("Repair engine" if state["managed"] else "Install engine")
        self.install_button.setEnabled(not state["error"] and not state["override"])
        self.remove_button.setEnabled(state["removable"] and not state["override"])
        self.existing_button.setEnabled(not state["override"])
        self.python.setEnabled(not state["override"])
        self.browse_button.setEnabled(not state["override"])
        self.cancel_button.setEnabled(False)
        self.gpu_button.setEnabled(True)
        self.close_button.setEnabled(True)
        if state["override"]:
            self.message.setText(
                "SAMSTARS_PYTHON controls this session. Unset it and restart QGIS "
                "to change the engine through setup."
            )
        else:
            self.message.setText(state["error"] or self._outcome or state.get("warning", ""))

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Python executable", self.python.text() or manager.engine_python() or "",
        )
        if path:
            self.python.setText(path)

    def _help(self):
        url = QUrl.fromLocalFile(str(Path(__file__).parent / "docs" / "index.html"))
        url.setFragment("setup")
        QDesktopServices.openUrl(url)

    def _install(self, checked=False, *, accelerator=None):
        python = self.python.text().strip() or None
        root = manager.engine_root()
        if accelerator is None:
            state = manager.status()
            accelerator = state.get("accelerator", "cpu") if state["managed"] else "cpu"
        self._start(lambda **kwargs: manager.install_engine(python, root, accelerator=accelerator, **kwargs))

    def _gpu_setup(self):
        state = manager.status()
        dialog = GpuSetupDialog(state, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        device = dialog.device.currentData()
        if dialog.operation == "test":
            self._start(
                lambda **kwargs: manager.check_gpu(state["python"], device, **kwargs),
                diagnosing=True,
            )
        else:
            self._install(accelerator=device)

    def _existing(self):
        python = self.python.text().strip()
        if not python:
            self.message.setText("Choose the Python executable of the existing environment first.")
            self.python.setFocus()
            return
        self._start(lambda **kwargs: manager.check_engine(python, **kwargs), checking=True)

    def _remove(self):
        answer = QMessageBox.question(
            self, "Remove SAMStars engine?",
            "Remove the private engines created by this plugin? Stop any running "
            "SAMStars jobs first. Existing external environments are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            root = manager.engine_root()
            self._start(lambda **kwargs: manager.remove_engine(root, **kwargs), removing=True)

    def _start(self, operation, *, removing=False, checking=False, diagnosing=False):
        if self._thread is not None:
            return
        self._removing = removing
        self._diagnosing = diagnosing
        self._outcome = ""
        self._successful = False
        self.log.clear()
        self.progress_bar.setRange(0, 0 if checking or diagnosing else 100)
        self.progress_bar.setValue(0)
        self.message.setText(
            "Testing the selected GPU engine…" if diagnosing else "Removing private engines…" if removing else
            "Checking the selected environment…" if checking else "Starting engine setup…"
        )
        for widget in (
            self.python, self.browse_button, self.install_button, self.remove_button,
            self.existing_button, self.close_button, self.gpu_button,
        ):
            widget.setEnabled(False)
        self.cancel_button.setEnabled(not removing)
        self._thread = SetupThread(operation, self)
        self._thread.progress.connect(self._progress)
        self._thread.message.connect(self.log.appendPlainText)
        self._thread.succeeded.connect(self._succeeded)
        self._thread.failed.connect(self._failed)
        self._thread.finished.connect(self._finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _progress(self, value, message):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(round(value))
        self.message.setText(message)
        self.log.appendPlainText(message)

    def _succeeded(self, result):
        if self._diagnosing:
            self._outcome = "PyTorch GPU test passed. " + (
                "StarDist refinement uses CPU." if result.get("refinement") == "cpu" else
                "TensorFlow refinement was not tested."
            )
            self.log.appendPlainText(result.get("hardware", ""))
            for check in result.get("checks", []):
                self.log.appendPlainText(check["name"] + ": " + check["version"])
            self.advanced.setCollapsed(False)
            self._successful = True
            return
        settings = QgsSettings()
        if "removed" in result:
            selected = settings.value(manager.SETTING, "", type=str)
            if selected and any(Path(selected).absolute().is_relative_to(Path(path)) for path in result["removed"]):
                settings.remove(manager.SETTING)
            self._outcome = "Private engines removed."
        else:
            settings.setValue(manager.SETTING, result["python"])
            self.python.clear()
            self._outcome = "Engine ready. Open Processing → Toolbox → SAMStars."
            if result.get("accelerator") in ("cuda", "mps"):
                name = "CUDA" if result["accelerator"] == "cuda" else "Apple MPS"
                self._outcome = (
                    f"GPU engine ready. Select {name} in the Processing tool's Advanced parameters. "
                    "StarDist refinement uses CPU."
                )
            self.log.appendPlainText("Validated engine: " + result["python"])
        self._successful = True
        self.completed.emit()

    def _failed(self, message, canceled):
        self._outcome = message.splitlines()[0][:240] if message else "Engine setup failed."
        self.message.setText(self._outcome)
        self.log.appendPlainText(message)
        if not canceled:
            self.advanced.setCollapsed(False)

    def _cancel(self):
        if self._thread is not None:
            if self._removing:
                self.message.setText("Finishing removal…")
                return
            self._thread.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.message.setText("Canceling setup…")

    def _finished(self):
        self._thread = None
        self._removing = False
        self._diagnosing = False
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if self._successful else 0)
        self._refresh()
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
