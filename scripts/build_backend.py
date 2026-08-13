from __future__ import annotations

import base64
import csv
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Any


def _metadata() -> str:
    return """Metadata-Version: 2.4
Name: model-optimization-rl-post-training-101
Version: 0.1.0
Summary: CPU-first post-training curriculum
Requires-Python: >=3.12
Provides-Extra: docs
Requires-Dist: mkdocs<2,>=1.6; extra == "docs"
Provides-Extra: dev
Requires-Dist: mkdocs<2,>=1.6; extra == "dev"
Requires-Dist: mypy<2,>=1.13; extra == "dev"
Requires-Dist: ruff<1,>=0.8; extra == "dev"
"""


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    del config_settings, metadata_directory
    root = Path(__file__).resolve().parents[1]
    name = "model_optimization_rl_post_training_101-0.1.0-py3-none-any.whl"
    output = Path(wheel_directory) / name
    dist_info = "model_optimization_rl_post_training_101-0.1.0.dist-info"
    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted((root / "src" / "pt101").glob("*.py")):
            target = f"pt101/{source.name}"
            payload = source.read_bytes()
            archive.writestr(target, payload)
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            )
            records.append((target, f"sha256={digest}", str(len(payload))))
        generated = {
            f"{dist_info}/METADATA": _metadata(),
            f"{dist_info}/WHEEL": "Wheel-Version: 1.0\nGenerator: pt101\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            f"{dist_info}/entry_points.txt": "[console_scripts]\npt101 = pt101.cli:main\n",
        }
        for target, text in generated.items():
            payload = text.encode()
            archive.writestr(target, payload)
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            )
            records.append((target, f"sha256={digest}", str(len(payload))))
        record_path = f"{dist_info}/RECORD"
        rows = [(*row,) for row in records] + [(record_path, "", "")]
        from io import StringIO

        stream = StringIO()
        csv.writer(stream, lineterminator="\n").writerows(rows)
        archive.writestr(record_path, stream.getvalue())
    return os.fspath(output.name)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build a wheel whose .pth points at the repository source directory."""
    del config_settings, metadata_directory
    root = Path(__file__).resolve().parents[1]
    name = "model_optimization_rl_post_training_101-0.1.0-py3-none-any.whl"
    output = Path(wheel_directory) / name
    dist_info = "model_optimization_rl_post_training_101-0.1.0.dist-info"
    files = {
        "model_optimization_rl_post_training_101.pth": f"{root / 'src'}\n".encode(),
        f"{dist_info}/METADATA": _metadata().encode(),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: pt101\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        f"{dist_info}/entry_points.txt": b"[console_scripts]\npt101 = pt101.cli:main\n",
    }
    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for target, payload in sorted(files.items()):
            archive.writestr(target, payload)
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            )
            records.append((target, f"sha256={digest}", str(len(payload))))
        record_path = f"{dist_info}/RECORD"
        from io import StringIO

        stream = StringIO()
        csv.writer(stream, lineterminator="\n").writerows(records + [(record_path, "", "")])
        archive.writestr(record_path, stream.getvalue())
    return os.fspath(output.name)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    del sdist_directory, config_settings
    raise RuntimeError("sdist is intentionally not implemented; use an editable install")
