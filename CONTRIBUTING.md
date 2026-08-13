# Contributing

Open an issue before a large curriculum change. Keep the canonical order beginner-safe, define jargon at first use, label evidence, cite primary sources, and add a deterministic practical or worked example when it changes understanding.

Edit `scripts/curriculum_data.py` for the stable spine or `scripts/research_data.py` for the dated research track. Research additions require a primary source, publication status, explicit evidence boundary, and a smallest-useful-scale reproduction. Then regenerate chapters and the book:

```bash
python3.12 scripts/generate_research.py
python3.12 scripts/generate_curriculum.py
python3.12 scripts/build_book.py
python3.12 scripts/validate_all.py
```

Do not add measured performance claims without raw artifacts and a named environment.
