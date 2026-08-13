from __future__ import annotations

import unittest
from typing import Any, cast

from scripts.generate_research import manifest
from scripts.research_data import RESEARCH_LESSONS, RESEARCH_SNAPSHOT_DATE


class ResearchTrackTests(unittest.TestCase):
    def test_track_contains_foundations_and_dated_frontier(self) -> None:
        slugs = {lesson.slug for lesson in RESEARCH_LESSONS}
        self.assertIn("ppo", slugs)
        self.assertIn("instructgpt", slugs)
        self.assertIn("dpo", slugs)
        self.assertIn("lora", slugs)
        self.assertIn("qlora", slugs)
        self.assertIn("constitutional-ai", slugs)
        self.assertIn("kto", slugs)
        self.assertIn("simpo", slugs)
        self.assertIn("rl-with-verifiable-physics", slugs)
        self.assertEqual(RESEARCH_SNAPSHOT_DATE, "2026-08-12")

    def test_track_includes_competing_capability_hypotheses(self) -> None:
        slugs = {lesson.slug for lesson in RESEARCH_LESSONS}
        self.assertIn("prorl", slugs)
        self.assertIn("power-sampling", slugs)
        self.assertIn("cross-domain-generalization", slugs)

    def test_manifest_labels_preprints_and_url_checks(self) -> None:
        snapshot = manifest()
        lessons = cast(list[dict[str, Any]], snapshot["lessons"])
        self.assertIsInstance(lessons, list)
        self.assertTrue(any("preprint" in lesson.status.lower() for lesson in RESEARCH_LESSONS))
        for record in lessons:
            self.assertEqual(record["url_check"]["paper_http_status"], 200)


if __name__ == "__main__":
    unittest.main()
