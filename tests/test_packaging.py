from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_backend import build_editable, build_wheel


class PackagingTests(unittest.TestCase):
    def test_regular_wheel_contains_package_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / build_wheel(directory)
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                self.assertIn("pt101/cli.py", names)
                entry = next(name for name in names if name.endswith("entry_points.txt"))
                self.assertIn("pt101 = pt101.cli:main", archive.read(entry).decode())
                metadata = next(name for name in names if name.endswith("METADATA"))
                metadata_text = archive.read(metadata).decode()
                self.assertIn("Provides-Extra: dev", metadata_text)
                self.assertIn('Requires-Dist: ruff<1,>=0.8; extra == "dev"', metadata_text)

    def test_editable_wheel_points_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / build_editable(directory)
            with zipfile.ZipFile(wheel) as archive:
                pointer = next(name for name in archive.namelist() if name.endswith(".pth"))
                self.assertTrue(archive.read(pointer).decode().strip().endswith("/src"))


if __name__ == "__main__":
    unittest.main()
