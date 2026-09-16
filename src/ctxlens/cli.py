"""ctxlens command-line interface."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import typer
from rich.console import Console

from ctxlens import __version__
from ctxlens.engine import analyze_file, analyze_text
from ctxlens.parsers import ParseError, available_formats
from ctxlens.reporters import (
    diff_to_dict,
    render_diff,
    render_html,
    render_terminal,
    to_json,
)
from ctxlens.reporters.json_report import to_dict
from ctxlens.tokenizers import available_tokenizers

app = typer.Typer(
    add_completion=False,
    help="ctxlens: a context-window profiler for AI agents.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

# exit codes
EXIT_OK = 0
EXIT_THRESHOLD = 2
EXIT_ERROR = 1


def _positive_int(value: int) -> int:
    if value < 1:
        raise typer.BadParameter("must be a positive integer (>= 1)")
    return value


def _non_negative_int(value: int) -> int:
    if value < 0:
        raise typer.BadParameter("must be a non-negative integer (>= 0)")
    return value


def _ratio(value: float | None) -> float | None:
    if value is None:
        return value
    if math.isnan(value) or math.isinf(value) or value < 0 or value > 1:
        raise typer.BadParameter("must be within 0-1")
    return value


def _version_cb(value: bool):
    if value:
        console.print(f"ctxlens {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_cb, is_eager=True, help="Show version and exit."
    ),
):
    """Profile what is consuming an AI agent's context window."""


@app.command()
def analyze(
    path: str = typer.Argument(..., help="Transcript file to analyze, or '-' to read stdin."),
    fmt: str = typer.Option("auto", "--format", "-f", help="Force a parser (auto detects)."),
    tokenizer: str = typer.Option("auto", "--tokenizer", "-t", help="Tokenizer: auto|heuristic|tiktoken."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a terminal report."),
    top: int = typer.Option(10, "--top", help="Number of top consumers to compute.", callback=_positive_int),
    tool_result_cap: int = typer.Option(400, "--tool-result-cap", help="Per tool-result token cap.", callback=_positive_int),
    tool_def_budget: int = typer.Option(800, "--tool-def-budget", help="Tool-definitions token budget.", callback=_non_negative_int),
    fail_over: float | None = typer.Option(
        None, "--fail-over-ratio", help="Exit non-zero if waste ratio exceeds this (0-1). CI-friendly.", callback=_ratio
    ),
):
    """Analyze a single transcript and print a context profile."""
    analysis = _load(path, fmt, tokenizer, top, tool_result_cap, tool_def_budget)

    if as_json:
        console.print_json(to_json(analysis))
    else:
        render_terminal(analysis, console)

    _maybe_fail(analysis.waste_ratio, fail_over)


@app.command()
def report(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Transcript file to analyze."),
    html: bool = typer.Option(False, "--html", help="Render a self-contained HTML report."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write report to this file."),
    fmt: str = typer.Option("auto", "--format", "-f"),
    tokenizer: str = typer.Option("auto", "--tokenizer", "-t"),
    tool_result_cap: int = typer.Option(400, "--tool-result-cap", callback=_positive_int),
    tool_def_budget: int = typer.Option(800, "--tool-def-budget", callback=_non_negative_int),
    fail_over: float | None = typer.Option(None, "--fail-over-ratio", callback=_ratio),
):
    """Generate an HTML (or JSON) report, to a file or stdout."""
    analysis = _load(path, fmt, tokenizer, 10, tool_result_cap, tool_def_budget)

    if html:
        content = render_html(analysis)
        default_ext = ".html"
    else:
        content = to_json(analysis)
        default_ext = ".json"

    if out is None and html:
        out = path.with_suffix(path.suffix + default_ext)

    if out is not None:
        try:
            out.write_text(content, encoding="utf-8")
        except OSError as exc:
            err_console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(EXIT_ERROR) from exc
        err_console.print(f"[green]Wrote[/green] {out}")
    else:
        sys.stdout.write(content + "\n")

    _maybe_fail(analysis.waste_ratio, fail_over)


@app.command()
def diff(
    a: Path = typer.Argument(..., exists=True, readable=True, help="First (baseline) transcript."),
    b: Path = typer.Argument(..., exists=True, readable=True, help="Second transcript."),
    tokenizer: str = typer.Option("auto", "--tokenizer", "-t"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON diff."),
):
    """Compare two sessions (baseline vs. candidate)."""
    an_a = _load(a, "auto", tokenizer, 10, 400, 800)
    an_b = _load(b, "auto", tokenizer, 10, 400, 800)
    if as_json:
        import json

        console.print_json(json.dumps(diff_to_dict(an_a, an_b)))
    else:
        render_diff(an_a, an_b, console)


@app.command()
def formats():
    """List supported transcript formats and available tokenizers."""
    console.print("[bold]Formats:[/bold] " + ", ".join(available_formats()))
    console.print("[bold]Tokenizers:[/bold] " + ", ".join(available_tokenizers()))


def _load(path, fmt, tokenizer, top, tool_result_cap, tool_def_budget):
    forced = None if fmt == "auto" else fmt
    common = {
        "fmt": forced,
        "tokenizer": tokenizer,
        "top_n": top,
        "tool_result_cap": tool_result_cap,
        "tool_def_budget": tool_def_budget,
    }
    try:
        if str(path) == "-":
            raw = sys.stdin.read()
            return analyze_text(raw, source="<stdin>", **common)
        p = Path(path)
        if not p.is_file():
            raise ParseError(f"no such file: {path}")
        return analyze_file(p, **common)
    except (ParseError, ImportError, ValueError, OSError) as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(EXIT_ERROR) from exc


def _maybe_fail(ratio: float, threshold: float | None):
    if threshold is not None and ratio >= threshold:
        err_console.print(
            f"[red]waste ratio {ratio * 100:.1f}% exceeds threshold {threshold * 100:.1f}%[/red]"
        )
        raise typer.Exit(EXIT_THRESHOLD)


# used by `to_dict` re-export for programmatic callers/tests
__all__ = ["app", "to_dict"]
