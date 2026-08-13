from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.curriculum_data import LESSONS, SOURCE_URLS  # noqa: E402
from scripts.generate_curriculum import chapter, mkdocs  # noqa: E402
from scripts.generate_curriculum import index as build_index  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_curriculum() -> None:
    manifest = json.loads((ROOT / "docs/reference/curriculum.json").read_text(encoding="utf-8"))
    require(manifest["lesson_count"] == len(LESSONS), "manifest lesson count is stale")
    require(
        manifest["order"] == [f"{i:02d}-{lesson.slug}" for i, lesson in enumerate(LESSONS)],
        "manifest order is stale",
    )
    require(len({lesson.slug for lesson in LESSONS}) == len(LESSONS), "lesson slugs must be unique")
    for index, lesson in enumerate(LESSONS):
        path = ROOT / "docs/spine" / f"{index:02d}-{lesson.slug}.md"
        require(path.exists(), f"missing {path.relative_to(ROOT)}")
        text = path.read_text(encoding="utf-8")
        require(text == chapter(index), f"{path.name} is stale; regenerate curriculum")
        for heading in (
            "The simplest accurate answer",
            "A useful mental model",
            "How it works",
            "Work one example",
            "Do it yourself",
            "Check your understanding",
            "Common failure",
            "Sources",
        ):
            require(f"## {heading}" in text, f"{path.name} misses {heading}")
        require("**Evidence in this chapter:**" in text, f"{path.name} misses evidence boundary")
        require(len(text.split()) >= 350, f"{path.name} is too thin")
    require(
        (ROOT / "docs/spine/index.md").read_text(encoding="utf-8") == build_index(),
        "spine index is stale",
    )
    require(
        (ROOT / "mkdocs.yml").read_text(encoding="utf-8") == mkdocs(),
        "mkdocs navigation is stale",
    )


def validate_links() -> None:
    markdown = (
        list((ROOT / "docs").rglob("*.md"))
        + list((ROOT / "book").glob("*.md"))
        + [ROOT / "README.md"]
    )
    pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for path in markdown:
        text = path.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            require(resolved.exists(), f"broken link in {path.relative_to(ROOT)}: {target}")


def validate_sources() -> None:
    used = {key for lesson in LESSONS for key in lesson.sources}
    require(
        used <= SOURCE_URLS.keys(), f"unregistered source keys: {sorted(used - SOURCE_URLS.keys())}"
    )
    require(len(used) >= 15, "source coverage is too narrow")


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, check=False)
    require(result.returncode == 0, f"command failed: {' '.join(command)}")


def main() -> int:
    validate_curriculum()
    validate_links()
    validate_sources()
    run([sys.executable, "scripts/build_book.py", "--check"])
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    print(f"validated {len(LESSONS)} ordered chapters, local links, sources, book, and tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
