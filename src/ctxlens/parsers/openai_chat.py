"""Parser for the generic OpenAI chat-messages format.

Accepts either a bare JSON array of messages::

    [{"role": "system", "content": "..."},
     {"role": "user", "content": "..."},
     {"role": "assistant", "content": "...", "tool_calls": [...]},
     {"role": "tool", "content": "..."}]

or an object ``{"messages": [...], "tools": [...]}`` where ``tools`` is the
function/tool schema list sent to the model.

Both ``system`` and ``developer`` messages contribute to the system segment.
"""

from __future__ import annotations

import json
from pathlib import Path

from ctxlens.models import Message, Segment, Session
from ctxlens.parsers.base import ParseError, Parser
from ctxlens.parsers.util import stringify, tool_ref


class OpenAIChatParser(Parser):
    format_name = "openai-chat"

    @classmethod
    def sniff(cls, raw: str, path: Path | None = None) -> float:
        raw = raw.strip()
        if not raw:
            return 0.0
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return 0.0
        messages = cls._messages_of(obj)
        if messages is None or not messages:
            return 0.0
        first = messages[0]
        if not isinstance(first, dict) or "role" not in first:
            return 0.0
        roles = {m.get("role") for m in messages if isinstance(m, dict)}
        score = 0.5
        if roles & {"system", "developer", "user", "assistant", "tool"}:
            score += 0.3
        # a bare array is the strongest generic signal
        if isinstance(obj, list):
            score += 0.1
        return min(score, 0.9)

    @staticmethod
    def _messages_of(obj):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            return obj["messages"]
        return None

    def parse(self, raw: str, path: Path | None = None) -> Session:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParseError(f"invalid JSON: {exc}") from exc

        messages = self._messages_of(obj)
        if messages is None:
            raise ParseError("expected a chat message array or {'messages': [...]}")

        out: list[Message] = []
        tools = obj.get("tools") if isinstance(obj, dict) else None
        if tools:
            out.append(
                Message(
                    Segment.TOOL_DEFINITIONS,
                    stringify(tools),
                    turn=1,
                    role="system",
                    meta={"tool_count": len(tools) if isinstance(tools, list) else None},
                )
            )

        turn = 1
        for entry in messages:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role", "user")
            content = entry.get("content")
            self._emit_message(entry, role, content, turn, out)
            if role == "assistant":
                turn += 1

        if not out:
            raise ParseError("no messages found in chat transcript")
        return Session(
            source_format=self.format_name,
            messages=out,
            meta={"path": str(path) if path else None},
        )

    def _emit_message(self, entry, role, content, turn, out):
        if role in {"system", "developer"}:
            out.append(Message(Segment.SYSTEM, stringify(content), turn, role=role))
            return
        if role == "tool":
            out.append(
                Message(
                    Segment.TOOL_RESULT,
                    stringify(content),
                    turn,
                    role=role,
                    ref=entry.get("tool_call_id"),
                    tool_name=entry.get("name"),
                )
            )
            return

        seg = Segment.ASSISTANT if role == "assistant" else Segment.USER
        if content:
            out.append(Message(seg, stringify(content), turn, role=role))

        for call in entry.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
            name = fn.get("name") or call.get("name")
            args_raw = fn.get("arguments", "")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) and args_raw else args_raw
            except json.JSONDecodeError:
                args = args_raw
            out.append(
                Message(
                    Segment.TOOL_CALL,
                    stringify(args),
                    turn,
                    role=role,
                    tool_name=name,
                    ref=tool_ref(name, args),
                    meta={"tool_call_id": call.get("id")},
                )
            )
