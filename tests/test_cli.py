"""Tests for the ``nev`` command-line interface."""

import json
import sqlite3

import pytest

from northstar_eval.cli import main
from conftest import SAMPLE_CSV, SAMPLE_DATASET


def test_cli_load_valid_dataset(capsys):
    assert main(["load", str(SAMPLE_DATASET)]) == 0
    out = capsys.readouterr().out
    assert "28" in out
    assert "Dataset is valid." in out


def test_cli_load_csv_dataset(capsys):
    assert main(["load", str(SAMPLE_CSV)]) == 0


def test_cli_load_invalid_dataset_returns_one(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a"}\n', encoding="utf-8")
    assert main(["load", str(bad)]) == 1
    assert "Error" in capsys.readouterr().err


def test_cli_run_writes_reports_and_db(tmp_path):
    db = tmp_path / "cli.db"
    out = tmp_path / "reports"
    assert main(["run", str(SAMPLE_DATASET), "--provider", "mock-b",
                 "--name", "cli test", "--db", str(db), "--out", str(out)]) == 0
    assert db.is_file()
    assert (out / f"{sqlite3.connect(db).execute('SELECT id FROM runs').fetchone()[0]}.md").is_file()
    assert (out / f"{sqlite3.connect(db).execute('SELECT id FROM runs').fetchone()[0]}.json").is_file()


def test_cli_run_unknown_provider_returns_one(tmp_path):
    # argparse rejects an unknown --provider choice with exit code 2.
    with pytest.raises(SystemExit) as exc:
        main(["run", str(SAMPLE_DATASET), "--provider", "gpt-99",
              "--db", str(tmp_path / "x.db")])
    assert exc.value.code == 2


def test_cli_compare_and_report(tmp_path, capsys):
    db = tmp_path / "cli.db"
    main(["run", str(SAMPLE_DATASET), "--provider", "mock-a", "--db", str(db)])
    main(["run", str(SAMPLE_DATASET), "--provider", "mock-b", "--db", str(db)])
    conn = sqlite3.connect(db)
    a, b = [row[0] for row in conn.execute("SELECT id FROM runs ORDER BY rowid")]
    assert main(["compare", a, b, "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "Regressed: 12" in out
    assert "Improved: 0" in out
    report_path = tmp_path / "report.md"
    assert main(["report", b, "--db", str(db), "--out", str(report_path)]) == 0
    assert "Failure examples" in report_path.read_text(encoding="utf-8")
    json_path = tmp_path / "report.json"
    assert main(["report", b, "--db", str(db), "--format", "json",
                 "--out", str(json_path)]) == 0
    assert json.loads(json_path.read_text(encoding="utf-8"))["overall"]["mean_score"] == 4.45


def test_cli_review_flow(tmp_path, capsys):
    db = tmp_path / "cli.db"
    main(["run", str(SAMPLE_DATASET), "--provider", "mock-b", "--db", str(db)])
    conn = sqlite3.connect(db)
    run_id = conn.execute("SELECT id FROM runs").fetchone()[0]
    assert main(["review", "list", "--run", run_id, "--db", str(db)]) == 0
    assert "12 item(s) awaiting review" in capsys.readouterr().out
    result_id = conn.execute(
        "SELECT id FROM results WHERE run_id = ? AND needs_review = 1 LIMIT 1",
        (run_id,),
    ).fetchone()[0]
    assert main(["review", "show", result_id, "--db", str(db)]) == 0
    assert main(["review", "judge", result_id, "--reviewer", "CLI Tester",
                 "--judgement", "agree", "--db", str(db)]) == 0
    assert "Review recorded: agree" in capsys.readouterr().out
    # Disagree without a corrected score must fail.
    assert main(["review", "judge", result_id, "--reviewer", "CLI Tester",
                 "--judgement", "disagree", "--db", str(db)]) == 1


def test_cli_version_and_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
