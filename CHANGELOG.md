# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Attribute OpenAI chat `developer` messages to the system segment instead of
  the user segment.

## [0.1.1] - 2026-08-06

### Changed

- Repository moved to the `AgentPostmortem` GitHub organization; project URLs now
  point at the new location. The package name is unchanged.

## [0.1.0] - 2026-07-27

### Added

- Transcript parsers with auto-detection for Claude Code session JSONL,
  OpenAI/Codex-style session JSON, and the generic OpenAI chat-messages format.
- Pluggable tokenizer layer: a deterministic, dependency-free heuristic
  tokenizer and an optional exact `tiktoken` backend (`auto` prefers tiktoken).
- Analysis: per-turn context composition by segment, cumulative growth and
  high-water mark, biggest single consumers, duplicate-content detection, and a
  waste report (duplicates, tool-result bloat, stale outputs, oversized tool
  definitions).
- Rule-based recommendations engine with severities and estimated savings.
- Reporters: rich terminal report with tables and a sparkline growth chart,
  `--json`, a self-contained HTML report with an inline SVG chart, and a
  session `diff`.
- CLI: `ctxlens analyze` (with stdin support via `-`), `report`, `diff`, and
  `formats`, plus CI-friendly exit codes and a `--fail-over-ratio` threshold.
