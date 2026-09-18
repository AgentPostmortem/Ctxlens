from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from ctxlens.cli import EXIT_THRESHOLD, _maybe_fail, app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ctxlens" in result.stdout


def test_formats_command():
    result = runner.invoke(app, ["formats"])
    assert result.exit_code == 0
    assert "claude-code-jsonl" in result.stdout


def test_analyze_terminal(claude_jsonl):
    result = runner.invoke(app, ["analyze", str(claude_jsonl)])
    assert result.exit_code == 0
    assert "ctxlens" in result.stdout


def test_analyze_json(claude_jsonl):
    result = runner.invoke(app, ["analyze", str(claude_jsonl), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["total_tokens"] > 0


def test_analyze_fail_over_threshold_exceeded(claude_jsonl):
    result = runner.invoke(
        app,
        ["analyze", str(claude_jsonl), "--tool-result-cap", "40", "--fail-over-ratio", "0.1"],
    )
    assert result.exit_code == 2


def test_analyze_fail_over_threshold_ok(claude_jsonl):
    result = runner.invoke(
        app, ["analyze", str(claude_jsonl), "--fail-over-ratio", "0.99"]
    )
    assert result.exit_code == 0


def test_maybe_fail_exact_threshold():
    with pytest.raises(typer.Exit) as exc_info:
        _maybe_fail(0.5, 0.5)
    assert exc_info.value.exit_code == EXIT_THRESHOLD


def test_maybe_fail_below_threshold():
    _maybe_fail(0.4, 0.5)
    _maybe_fail(0.5, None)


def test_report_html_to_file(tmp_path, codex_session):
    out = tmp_path / "r.html"
    result = runner.invoke(app, ["report", str(codex_session), "--html", "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert out.read_text().startswith("<!doctype html>")


def test_report_missing_parent_dir(tmp_path, codex_session):
    out = tmp_path / "missing" / "dir" / "r.html"
    result = runner.invoke(app, ["report", str(codex_session), "--html", "-o", str(out)])
    assert result.exit_code == 1
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_diff_json(openai_array, openai_chat):
    result = runner.invoke(
        app, ["diff", str(openai_array), str(openai_chat), "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "delta_tokens" in data


def test_analyze_missing_file():
    result = runner.invoke(app, ["analyze", "/no/such/file.jsonl"])
    assert result.exit_code == 1


@pytest.mark.parametrize(
    ("args", "reason"),
    [
        (["--top", "-1"], "must be a positive integer (>= 1)"),
        (["--top", "0"], "must be a positive integer (>= 1)"),
        (["--tool-result-cap", "-5"], "must be a positive integer (>= 1)"),
        (["--tool-def-budget", "-1"], "must be a non-negative integer (>= 0)"),
        (["--fail-over-ratio", "5"], "must be within 0-1"),
        (["--fail-over-ratio", "-0.5"], "must be within 0-1"),
        (["--fail-over-ratio", "nan"], "must be within 0-1"),
    ],
)
def test_analyze_rejects_out_of_range_numeric_options(claude_jsonl, args, reason):
    result = runner.invoke(app, ["analyze", str(claude_jsonl), *args])
    assert result.exit_code == 2
    assert "Invalid value" in result.stderr
    assert reason in result.stderr


def test_analyze_stdin(openai_array):
    raw = openai_array.read_text()
    result = runner.invoke(app, ["analyze", "-", "--json"], input=raw)
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["source"] == "<stdin>"
    assert data["total_tokens"] > 0
