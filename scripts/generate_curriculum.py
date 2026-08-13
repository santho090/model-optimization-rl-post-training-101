from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.curriculum_data import LESSONS, SOURCE_URLS  # noqa: E402
from scripts.research_data import RESEARCH_LESSONS  # noqa: E402


def chapter(index: int) -> str:
    lesson = LESSONS[index]
    previous = (
        "None. Start here."
        if index == 0
        else f"[Chapter {index - 1:02d}](../spine/{index - 1:02d}-{LESSONS[index - 1].slug}.md)"
    )
    next_link = (
        "Proceed to the capstone packet."
        if index == len(LESSONS) - 1
        else f"[Chapter {index + 1:02d}](../spine/{index + 1:02d}-{LESSONS[index + 1].slug}.md)"
    )
    source_lines = "\n".join(
        f"- [{SOURCE_URLS[key][0]}]({SOURCE_URLS[key][1]})" for key in lesson.sources
    )
    return f"""# {index:02d}. {lesson.title}

**Question:** {lesson.question}

**Evidence in this chapter:** stable concepts are explained from cited primary sources. All `pt101` outputs are deterministic `simulated` teaching evidence, not measurements of an LLM.

## The simplest accurate answer

{lesson.core}

## A useful mental model

{lesson.analogy}

## How it works

{lesson.mechanism}

```mermaid
flowchart LR
    A[task and frozen evaluation] --> B[data or environment]
    B --> C[policy produces logits or samples]
    C --> D[loss, preference, or reward]
    D --> E[gradient update]
    E --> F[candidate evaluation]
    F -->|all gates pass| G[promote]
    F -->|any blocker fails| H[reject and diagnose]
```

## Work one example

{lesson.example}

## Do it yourself

{lesson.lab}

Record the input, configuration, output, evidence label, and one sentence stating what the output **does not** prove. If you change two variables, split the work into two experiments.

## Check your understanding

{lesson.check}

## Common failure

{lesson.trap}

## Sources

{source_lines}

## Course position

- Prerequisite: {previous}
- Next: {next_link}
"""


def index() -> str:
    rows = "\n".join(
        f"| {i:02d} | [{lesson.title}]({i:02d}-{lesson.slug}.md) | {lesson.question} |"
        for i, lesson in enumerate(LESSONS)
    )
    return f"""# The canonical beginner path

Read these chapters in order. Each chapter introduces only the machinery needed by the next one. The course uses one continuous teaching loop so that SFT, reward modeling, DPO, PPO, and GRPO are not isolated vocabulary.

```text
task contract
  -> frozen baseline evaluation
  -> demonstrations and SFT
  -> preferences and reward models
  -> choose offline preference optimization or online RL
  -> distributed execution with artifact identities
  -> independent gates
  -> staged promotion or rejection
```

| # | Chapter | Guiding question |
| ---: | --- | --- |
{rows}

The first run needs Python 3.12 and no GPU, external model, or network after installation. Read [the evidence contract](../reference/evidence.md) before interpreting output.
"""


def mkdocs() -> str:
    nav = "\n".join(
        f'      - "{i:02d} {lesson.title}": spine/{i:02d}-{lesson.slug}.md'
        for i, lesson in enumerate(LESSONS)
    )
    research_nav = "\n".join(
        f'      - "R{i:02d} {lesson.title}": research/r{i:02d}-{lesson.slug}.md'
        for i, lesson in enumerate(RESEARCH_LESSONS)
    )
    return f"""site_name: model optimization, RL, and post-training 101
site_description: A CPU-first course from gradients to safe post-training promotion.
docs_dir: docs
theme:
  name: mkdocs
nav:
  - start here: index.md
  - canonical beginner path:
      - course map: spine/index.md
{nav}
  - important and current research:
      - research route: research/index.md
{research_nav}
  - practicals:
      - playgrounds: playgrounds/index.md
      - research map: playgrounds/research-map.html
      - end-to-end capstone: capstones/end-to-end.md
  - reference:
      - evidence contract: reference/evidence.md
      - algorithm chooser: reference/algorithm-chooser.md
      - real training bridge: reference/real-training-bridge.md
      - glossary: reference/glossary.md
      - sources: reference/sources.md
      - experiment record: reference/experiment-record.md
markdown_extensions:
  - tables
  - fenced_code
"""


def main() -> int:
    spine = ROOT / "docs" / "spine"
    spine.mkdir(parents=True, exist_ok=True)
    expected: set[Path] = set()
    for i, _lesson in enumerate(LESSONS):
        path = spine / f"{i:02d}-{_lesson.slug}.md"
        path.write_text(chapter(i), encoding="utf-8")
        expected.add(path)
    for existing in spine.glob("[0-9][0-9]-*.md"):
        if existing not in expected:
            existing.unlink()
    (spine / "index.md").write_text(index(), encoding="utf-8")
    (ROOT / "mkdocs.yml").write_text(mkdocs(), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "order": [f"{i:02d}-{lesson.slug}" for i, lesson in enumerate(LESSONS)],
        "lesson_count": len(LESSONS),
        "evidence_boundary": "Generated chapters plus deterministic CPU teaching code; no LLM training run is claimed.",
    }
    (ROOT / "docs" / "reference" / "curriculum.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
