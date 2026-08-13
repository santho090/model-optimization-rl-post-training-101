# Contributing

Open an issue before a large curriculum change. Keep the canonical order beginner-safe, define jargon at first use, label evidence, cite primary sources, and add a deterministic practical or worked example when it changes understanding.

Edit `scripts/curriculum_data.py`, regenerate chapters and the book, then run:

```bash
python3.12 scripts/generate_curriculum.py
python3.12 scripts/build_book.py
python3.12 scripts/validate_all.py
```

Do not add measured performance claims without raw artifacts and a named environment.
