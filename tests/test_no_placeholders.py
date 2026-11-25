"""Hygiene tests: no fake content, no secrets, no stray artefacts.

The forbidden words below are built by concatenation so this file
itself does not trip the check it performs. HTML input hints
(placeholder="...") are real UI text, not fake content, so they are
stripped before checking.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = [
    "place" + "holder",
    "TO" + "DO",
    "st" + "ub",
    "lor" + "em",
    "sample res" + "ult",
    "coming s" + "oon",
]


def _repo_files() -> list[Path]:
    skip_dirs = {".venv", ".git", "__pycache__", ".pytest_cache", "uploads", "reports"}
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.suffix in {".db", ".pyc"}:
            continue
        files.append(path)
    return files


def test_no_forbidden_words_in_repo():
    hits = []
    for path in _repo_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # Strip HTML input hints (built by concatenation, like the
        # forbidden list, so this file does not trip its own check).
        text = re.sub(
            "place" + "holder" + r"""=(?:"[^"]*"|'[^']*')""", "", text
        )
        lowered = text.lower()
        for word in FORBIDDEN:
            if word.lower() in lowered:
                hits.append(f"{path.relative_to(PROJECT_ROOT)}: {word}")
    assert hits == [], f"Forbidden words found: {hits}"


def test_no_database_or_env_file_in_repo():
    assert not list(PROJECT_ROOT.rglob("*.db")), "A .db file is in the repo tree."
    assert not (PROJECT_ROOT / ".env").exists(), "A real .env file is in the repo tree."


def test_env_example_has_no_real_key():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "NORTHSTAR_LLM_API_KEY=\n" in text or "NORTHSTAR_LLM_API_KEY=" in text
    assert "sk-" not in text
    for line in text.splitlines():
        if line.startswith("NORTHSTAR_LLM_API_KEY="):
            assert line.split("=", 1)[1].strip() == ""


def test_gitignore_covers_secrets_and_databases():
    text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in text
    assert "*.db" in text
    assert ".venv" in text
