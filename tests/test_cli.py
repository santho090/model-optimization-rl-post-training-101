from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pt101.cli import main


class CliTests(unittest.TestCase):
    def test_pipeline_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.json"
            self.assertEqual(main(["pipeline", "--output", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["evidence"], "simulated")


if __name__ == "__main__":
    unittest.main()
