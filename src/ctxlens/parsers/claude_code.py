"""Parser for Claude Code session transcripts (JSONL).

Claude Code writes one JSON object per line under
``~/.claude/projects/<slug>/<session>.jsonl``. Each line typically has a
``type`` ("user", "assistant", "system", "summary") and a ``message`` object
with ``role`` and ``content``. ``content`` is either a plain string or a list
of Anthropic content blocks (text, thinking, tool_use, tool_result).
"""

from __future__ import annotations

import json
from pathlib import Path

from ctxlens.models import Message, Segment, Session
from ctxlens.parsers.base import ParseError, Parser
from ctxlens.parsers.util import block_to_message


class ClaudeCodeParser(Parser):
    format_name = "claude-code-jsonl"

    @classmethod
    def sniff(cls, raw: str, path: Path | None = None) -> float:
        raw = raw.strip()
        if not raw:
            return 0.0
        # a JSONL transcript has multiple JSON objects, one per line; scan the
        # first few so a leading "summary" line doesn't hide the signal
        best = 0.0
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) < 2 and not raw.startswith("{"):
            return 0.0
        for line in lines[:5]:
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return 0.0
            if not isinstance(obj, dict):
                continue
            score = 0.0
            for key in ("uuid", "parentUuid", "sessionId", "cwd"):
                if key in obj:
                    score += 0.2
            if obj.get("type") in {"user", "assistant", "system", "summary"} and (
                "message" in obj or obj.get("type") in {"summary", "system"}
            ):
                score += 0.4
            best = max(best, min(score, 1.0))
        return best

    def parse(self, raw: str, path: Path | None = None) -> Session:
        messages: list[Message] = []
        turn = 1
        meta: dict = {}
        saw_content = False

        for lineno, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseError(f"invalid JSON on line {lineno}: {exc}") from exc
            if not isinstance(entry, dict):
                continue

            etype = entry.get("type")
            if etype == "summary":
                summary = entry.get("summary")
                if summary:
                    existing = meta.get("summary")
                    meta["summary"] = (
                        f"{existing}\n{summary}" if existing else str(summary)
                    )
                continue
            if etype == "system" and "message" not in entry:
                text = entry.get("content") or entry.get("text") or ""
                if text:
                    messages.append(Message(Segment.SYSTEM, str(text), turn, role="system"))
                    saw_content = True
                continue

            msg = entry.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", etype or "user")
            content = msg.get("content")

            produced = self._emit(content, role, turn, messages)
            saw_content = saw_content or produced

            if role == "assistant" and produced:
                turn += 1

        if not saw_content:
            raise ParseError("no messages found in Claude Code transcript")

        meta.setdefault("path", str(path) if path else None)
        return Session(source_format=self.format_name, messages=messages, meta=meta)

    def _emit(self, content, role: str, turn: int, out: list[Message]) -> bool:
        produced = False
        if isinstance(content, str):
            if content.strip():
                seg = Segment.ASSISTANT if role == "assistant" else Segment.USER
                out.append(Message(seg, content, turn, role=role))
                produced = True
            return produced
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                m = block_to_message(block, role=role, turn=turn)
                if m is not None and (m.text or m.segment == Segment.TOOL_RESULT):
                    out.append(m)
                    produced = True
        return produced
