"""Parser for OpenAI/Codex-style session JSON.

Codex-style rollouts store a session object plus an ``items`` list using the
Responses API item shapes::

    {
      "session": {"id": "...", "instructions": "<system prompt>"},
      "tools": [ ... ],
      "items": [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "..."}]},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "..."}]},
        {"type": "reasoning", "summary": [{"text": "..."}]},
        {"type": "function_call", "name": "shell",
         "arguments": "{...}", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "..."}
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from ctxlens.models import Message, Segment, Session
from ctxlens.parsers.base import ParseError, Parser
from ctxlens.parsers.util import stringify, tool_ref


class CodexParser(Parser):
    format_name = "codex-session"

    @classmethod
    def sniff(cls, raw: str, path: Path | None = None) -> float:
        raw = raw.strip()
        if not raw:
            return 0.0
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return 0.0
        if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
            return 0.0
        score = 0.6
        if "session" in obj or "instructions" in obj:
            score += 0.2
        item_types = {i.get("type") for i in obj["items"] if isinstance(i, dict)}
        if item_types & {"function_call", "function_call_output", "message", "reasoning"}:
            score += 0.2
        return min(score, 1.0)

    def parse(self, raw: str, path: Path | None = None) -> Session:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc}") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
            raise ParseError("expected a Codex session object with an 'items' list")

        out: list[Message] = []
        turn = 1
        session = obj.get("session") if isinstance(obj.get("session"), dict) else {}
        instructions = obj.get("instructions") or session.get("instructions")
        if instructions:
            out.append(Message(Segment.SYSTEM, stringify(instructions), turn, role="system"))
        if obj.get("tools"):
            tools = obj["tools"]
            out.append(
                Message(
                    Segment.TOOL_DEFINITIONS,
                    stringify(tools),
                    turn,
                    role="system",
                    meta={"tool_count": len(tools) if isinstance(tools, list) else None},
                )
            )

        for item in obj["items"]:
            if not isinstance(item, dict):
                continue
            turn = self._emit_item(item, turn, out)

        if not out:
            raise ParseError("no messages found in Codex session")
        meta = {"path": str(path) if path else None, "session_id": session.get("id")}
        return Session(source_format=self.format_name, messages=out, meta=meta)

    def _emit_item(self, item: dict, turn: int, out: list[Message]) -> int:
        itype = item.get("type")
        if itype == "message":
            role = item.get("role", "user")
            text = self._content_text(item.get("content"))
            if not text.strip():
                return turn
            seg = Segment.ASSISTANT if role == "assistant" else Segment.USER
            if role == "system":
                seg = Segment.SYSTEM
            out.append(Message(seg, text, turn, role=role))
            if role == "assistant":
                return turn + 1
            return turn
        if itype == "reasoning":
            text = self._reasoning_text(item)
            out.append(Message(Segment.THINKING, text, turn, role="assistant"))
            return turn
        if itype == "function_call":
            name = item.get("name")
            args_raw = item.get("arguments", "")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else args_raw
            except json.JSONDecodeError:
                args = args_raw
            out.append(
                Message(
                    Segment.TOOL_CALL,
                    stringify(args),
                    turn,
                    role="assistant",
                    tool_name=name,
                    ref=tool_ref(name, args),
                    meta={"call_id": item.get("call_id")},
                )
            )
            return turn
        if itype == "function_call_output":
            out.append(
                Message(
                    Segment.TOOL_RESULT,
                    stringify(item.get("output")),
                    turn,
                    role="tool",
                    ref=item.get("call_id"),
                )
            )
            return turn
        return turn

    @staticmethod
    def _content_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", "") or stringify(block.get("content", "")))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return stringify(content)

    @staticmethod
    def _reasoning_text(item: dict) -> str:
        summary = item.get("summary")
        if isinstance(summary, list):
            return "".join(
                b.get("text", "") for b in summary if isinstance(b, dict)
            )
        return stringify(summary or item.get("text", ""))
