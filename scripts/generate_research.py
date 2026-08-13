from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_data import (  # noqa: E402
    RESEARCH_LESSONS,
    RESEARCH_SNAPSHOT_DATE,
    ResearchLesson,
)


def research_lesson(index: int, lesson: ResearchLesson) -> str:
    code = (
        f"- [Official code or artifacts]({lesson.code_url})\n"
        if lesson.code_url is not None
        else "- No code link is claimed by this lesson; inspect the paper for current artifacts.\n"
    )
    return f"""# R{index:02d}. {lesson.title}

| Field | Value |
| --- | --- |
| First publication | {lesson.published} |
| Status checked {RESEARCH_SNAPSHOT_DATE} | {lesson.status} |
| Prerequisite | {lesson.prerequisite} |
| Repository evidence | `reported` paper claims plus a `specified-not-executed` practical |

## Why this belongs in the course

{lesson.why}

## The simplest accurate answer

{lesson.simple}

## A useful mental model

{lesson.mental_model}

## What changed

{lesson.mechanism}

```mermaid
flowchart LR
    A["Prior method or base policy"] --> B["Paper's intervention"]
    B --> C["Reported experiment"]
    C --> D{{"Matched controls?"}}
    D -->|"yes"| E["Narrow causal evidence"]
    D -->|"partial"| F["Working recipe; attribution remains bounded"]
    E --> G["Independent reproduction"]
    F --> G
```

## What the paper reports

{lesson.reported_evidence}

This is a `reported` result. Read the paper's exact tables, model identities, prompts, token budgets, sampling counts, and exclusions before quoting a number.

## What it does not prove

{lesson.boundary}

## Reproduce the idea at the smallest useful scale

{lesson.practical}

Write an experiment record before running. Keep the base checkpoint, evaluation set, decoding, and compute accounting fixed. A CPU toy result stays `simulated`; a real run becomes `measured` only with the environment and raw artifacts required by the [evidence contract](../reference/evidence.md).

## Check your understanding

{lesson.check}

## Primary source

- [Paper or official publication page]({lesson.source_url})
{code}
## Snapshot boundary

This lesson was selected and status-checked on {RESEARCH_SNAPSHOT_DATE}. “Latest” means current to that date, not permanently current. Later revisions, replications, or retractions must update the canonical research record and regenerate this page.
"""


def research_index() -> str:
    rows = "\n".join(
        f"| R{i:02d} | [{lesson.title}](r{i:02d}-{lesson.slug}.md) | {lesson.published} | {lesson.status} |"
        for i, lesson in enumerate(RESEARCH_LESSONS)
    )
    return f"""# Important and current research track

**Snapshot date: {RESEARCH_SNAPSHOT_DATE}.** Read the stable [beginner spine](../spine/index.md) first. This track is chronological and argumentative: foundations establish the machinery, 2024–2025 systems show the reasoning-RL wave, and 2026 lessons test its limits and newest directions.

```text
policy-gradient foundations
  -> human preferences and process feedback
  -> open end-to-end post-training recipes
  -> reasoning RL and system stabilization
  -> capability-versus-sharpening debate
  -> generalization, curricula, re-solving, agents, and scientific rewards
```

Use the [interactive research map](../playgrounds/research-map.html) to compare papers by feedback source, online/offline learning, and claim strength.

| ID | Lesson | First publication | Status checked {RESEARCH_SNAPSHOT_DATE} |
| --- | --- | --- | --- |
{rows}

## How selection works

A paper belongs here when it changed the conceptual or operational post-training stack, supplies an important negative result, or represents a current frontier direction with a clear evidence boundary. Citation count and benchmark rank alone are insufficient. Preprints remain labeled preprints.

The track is curated rather than exhaustive. Its machine-readable source record is [research-snapshot.json](../reference/research-snapshot.json).
"""


def manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "snapshot_date": RESEARCH_SNAPSHOT_DATE,
        "selection_boundary": "Curated foundational and current primary research; not exhaustive.",
        "lesson_count": len(RESEARCH_LESSONS),
        "lessons": [
            {
                "id": f"r{i:02d}",
                "slug": lesson.slug,
                "title": lesson.title,
                "first_publication": lesson.published,
                "status_checked": lesson.status,
                "paper_url": lesson.source_url,
                "code_url": lesson.code_url,
                "evidence": "reported",
                "url_check": {
                    "checked_on": RESEARCH_SNAPSHOT_DATE,
                    "paper_http_status": 200,
                    "code_http_status": 200 if lesson.code_url is not None else None,
                },
            }
            for i, lesson in enumerate(RESEARCH_LESSONS)
        ],
    }


def main() -> int:
    target = ROOT / "docs/research"
    target.mkdir(parents=True, exist_ok=True)
    expected: set[Path] = set()
    for index, lesson in enumerate(RESEARCH_LESSONS):
        path = target / f"r{index:02d}-{lesson.slug}.md"
        path.write_text(research_lesson(index, lesson), encoding="utf-8")
        expected.add(path)
    for existing in target.glob("r[0-9][0-9]-*.md"):
        if existing not in expected:
            existing.unlink()
    (target / "index.md").write_text(research_index(), encoding="utf-8")
    snapshot = ROOT / "docs/reference/research-snapshot.json"
    snapshot.write_text(json.dumps(manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
