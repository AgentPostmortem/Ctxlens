# ctxlens

**A flamegraph for your agent's context window.** ctxlens parses AI agent
session transcripts and shows you exactly what is eating your context per turn,
flags the waste, and tells you what to cut.

[![CI](https://github.com/AgentPostmortem/Ctxlens/actions/workflows/ci.yml/badge.svg)](https://github.com/AgentPostmortem/Ctxlens/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Why

Agents get slow, expensive, and dumb when their context window fills with junk:
the same file read six times, a 12k-token tool result that mattered for one
turn, tool schemas re-sent on every step. Token dashboards tell you the bill.
**ctxlens tells you where the bytes went and what to delete.**

It works offline with a deterministic heuristic tokenizer (no network, no heavy
deps), and upgrades to exact counts automatically when `tiktoken` is installed.

## Quickstart

**PyPI:** https://pypi.org/project/ctxlens-cli/

```bash
pip install ctxlens-cli                # core
pip install "ctxlens-cli[tiktoken]"    # optional exact token counts

# From a clone, use the shipped fixture (no real session required):
ctxlens analyze tests/fixtures/claude_code_session.jsonl
```

Expected shape (numbers vary with tokenizer; structure is stable):

```text
Source   tests/fixtures/claude_code_session.jsonl
Format   claude-code-jsonl
Tokens   …   Turns …   High-water …
Waste    … tokens (…%)
```

```bash
ctxlens report tests/fixtures/claude_code_session.jsonl --html -o report.html
ctxlens diff before.jsonl after.jsonl
```

Point it at a real Claude Code session, or pipe a transcript in:

```bash
ctxlens analyze ~/.claude/projects/<slug>/<session>.jsonl
cat session.json | ctxlens analyze - --json
```

## Example output

```
╭─ ctxlens ────────────────────────────────────╮
│ Source   session.jsonl                        │
│ Format   claude-code-jsonl   Tokenizer heuristic │
│ Tokens   12,481   Turns 14   High-water 12,481 │
│ Waste    4,932 tokens (39.5%)                 │
╰───────────────────────────────────────────────╯
Context composition by segment
 Segment       Tokens     %  Msgs  Share
 tool result    6,204  49.7    22  ██████████████·······
 assistant      2,110  16.9    14  ██████···············
 system         1,540  12.3     1  ████·················
 ...
Context growth
 cumulative  ▁▁▂▂▃▄▄▅▆▆▇▇██  peak 12,481
 per-turn    ▂█▃▂▅▂▁▂▇▂▁▃▂▁  max 3,204
Recommendations
 [HIGH] Repeated content wastes tokens  (~2,410 tok)
     'Read:file_path=config.py' appears 6 times (turns [2, 5, 7, 9, 11, 13]) ...
```

<!-- HTML report screenshot: docs/report.png -->

## Supported formats

ctxlens auto-detects the format; override with `--format`.

| Format | `--format` | Source |
| --- | --- | --- |
| Claude Code session | `claude-code-jsonl` | `~/.claude/projects/*/*.jsonl` |
| OpenAI/Codex session | `codex-session` | Codex rollout JSON (`items[]`) |
| Generic OpenAI chat | `openai-chat` | chat-messages array or `{messages, tools}` |

Every message is attributed to a segment: `system`, `tool_definitions`,
`user`, `assistant`, `thinking`, `tool_call`, `tool_result`.

## Recommendations explained

The engine is rule-based and specific, not generic advice:

- **Tool results dominate** — flagged when tool outputs are >=30% of context.
- **Repeated content** — the same file/tool result (by reference or by exact
  body) appearing more than once; every copy after the first is wasted.
- **Stale tool outputs** — an older result superseded by a newer one for the
  same target still occupies context.
- **Oversized tool definitions** — tool schemas above budget, paid every turn.
- **System prompt weight** and **single biggest consumer** callouts.

Each recommendation carries a severity and an estimated token saving.

## Waste report

`waste_ratio = total_waste / total_tokens`, where total waste sums duplicate
tokens, tool-result bloat (tokens above `--tool-result-cap`), stale tool
outputs, and tool-definition overage (above `--tool-def-budget`).

## CI usage

Fail a build when a captured agent session wastes too much context:

```bash
ctxlens analyze session.jsonl --fail-over-ratio 0.30
```

Exit codes: `0` OK, `2` threshold exceeded, `1` error. Emit machine-readable
output with `--json`, or diff a baseline against a candidate in CI with
`ctxlens diff baseline.jsonl candidate.jsonl --json`.

## Tokenizers

- `heuristic` (default fallback): deterministic, dependency-free, great for
  relative profiling and CI.
- `tiktoken`: exact BPE counts when installed. `--tokenizer auto` prefers it.

## Contributing

```bash
git clone https://github.com/AgentPostmortem/Ctxlens
cd ctxlens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest
```

Issues and PRs welcome. Adding a parser? Implement `Parser.sniff` and
`Parser.parse`, register it in `ctxlens/parsers/__init__.py`, and drop a
fixture in `tests/fixtures/`.

## License

MIT © royalpinto007
