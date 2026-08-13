from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.curriculum_data import LESSONS  # noqa: E402
from scripts.research_data import RESEARCH_LESSONS  # noqa: E402


def build() -> str:
    parts = [
        "# Model optimization, RL, and post-training 101\n\n",
        "A single-file edition of the canonical beginner path. Links to repository reference pages remain relative to `book/`.\n\n",
    ]
    for index, lesson in enumerate(LESSONS):
        text = (ROOT / "docs" / "spine" / f"{index:02d}-{lesson.slug}.md").read_text(
            encoding="utf-8"
        )
        text = text.replace(
            f"# {index:02d}.", f'<a id="chapter-{index:02d}"></a>\n\n## {index:02d}.', 1
        )
        text = text.replace("(../spine/", "(../docs/spine/").replace(
            "(../reference/", "(../docs/reference/"
        )
        parts.extend([text, "\n\n---\n\n"])
    parts.extend(
        [
            '<a id="research-track"></a>\n\n# Important and current research track\n\n',
            "The following lessons are a dated research appendix. Their paper results are reported evidence, not local reproductions.\n\n",
        ]
    )
    for index, research_item in enumerate(RESEARCH_LESSONS):
        path = ROOT / "docs/research" / f"r{index:02d}-{research_item.slug}.md"
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            f"# R{index:02d}.",
            f'<a id="research-r{index:02d}"></a>\n\n## R{index:02d}.',
            1,
        )
        text = text.replace("(../reference/", "(../docs/reference/")
        parts.extend([text, "\n\n---\n\n"])
    return "".join(parts).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = ROOT / "book" / "model-optimization-rl-post-training-101.md"
    expected = build()
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != expected:
            raise SystemExit("book is stale; run python3.12 scripts/build_book.py")
        return 0
    output.parent.mkdir(exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
