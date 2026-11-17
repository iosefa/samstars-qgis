"""An offline environment check with an ordinary Processing HTML result."""

from html import escape
from pathlib import Path

from qgis.core import QgsProcessingException, QgsProcessingOutputBoolean, QgsProcessingParameterFileDestination

from .common import SamStarsAlgorithm
from ..runtime.client import EngineCanceled, EngineError, call_engine


class DependencyCheckAlgorithm(SamStarsAlgorithm):
    def name(self):
        return "checkdependencies"

    def displayName(self):
        return "Check SAMStars dependencies"

    def group(self):
        return "Setup"

    def groupId(self):
        return "setup"

    def shortHelpString(self):
        return (
            "Check the configured Python engine, SAMStars version, scientific imports, and small "
            "CPU, raster, and GeoPackage operations. No models are downloaded. A successful check does not validate a "
            "trained model or GPU drivers. Use Plugins → SAMStars → Set up SAMStars to install "
            "a private engine or select an existing environment. GPU setup → Test selected engine checks "
            "PyTorch GPU computation separately; private engines keep StarDist refinement on CPU."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFileDestination("REPORT", "Dependency report", "HTML files (*.html)"))
        self.addOutput(QgsProcessingOutputBoolean("READY", "Engine ready"))

    def processAlgorithm(self, parameters, context, feedback):
        try:
            result = call_engine("check", {}, feedback)
        except EngineCanceled as exc:
            raise QgsProcessingException(str(exc)) from exc
        except EngineError as exc:
            result = {"ready": False, "checks": [{"name": "Engine", "ok": False, "version": "", "error": str(exc)}]}
        lines = [
            f"{'OK' if item['ok'] else 'FAIL'}: {item['name']} {item['version']} {item['error']}".strip()
            for item in result["checks"]
        ]
        status = "Ready" if result["ready"] else "Setup required"
        for line in lines:
            feedback.pushInfo(line)
        if not result["ready"]:
            feedback.reportError("Open Plugins → SAMStars → Set up SAMStars.")
        report = Path(self.parameterAsFileOutput(parameters, "REPORT", context))
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>SAMStars dependencies</title>'
            f"<h1>SAMStars: {status}</h1><pre>{escape(chr(10).join(lines))}</pre></html>",
            encoding="utf-8",
        )
        return {"REPORT": str(report), "READY": result["ready"]}
