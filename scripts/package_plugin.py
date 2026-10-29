#!/usr/bin/env python3
"""Build the QGIS-installable ZIP without dependencies or build tools."""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ID = "samstars_qgis"


def package(destination=None):
    destination = Path(destination) if destination else ROOT / "dist" / f"{PLUGIN_ID}.zip"
    files = [ROOT / name for name in (
        "__init__.py", "metadata.txt", "samstars_plugin.py", "provider.py",
        "setup_dialog.py", "LICENSE", "README.md", "NOTICE",
    )]
    for directory in ("algorithms", "runtime", "icons", "docs"):
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Required plugin file missing: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(Path(PLUGIN_ID) / path.relative_to(ROOT)))
    return destination


if __name__ == "__main__":
    print(package())
