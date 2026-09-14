from __future__ import annotations

import json

import pytest

from ctxlens.models import Segment
from ctxlens.parsers import (
    ClaudeCodeParser,
    CodexParser,
    OpenAIChatParser,
    ParseError,
    detect_parser,
    parse_file,
    parse_text,
)


def _segments(session):
    return {m.segment for m in session.messages}


def test_detect_claude_code(claude_jsonl):
    raw = claude_jsonl.read_text()
    assert detect_parser(raw, claude_jsonl) is ClaudeCodeParser


def test_detect_codex(codex_session):
    raw = codex_session.read_text()
    assert detect_parser(raw, codex_session) is CodexParser


def test_detect_openai_array(openai_array):
    raw = openai_array.read_text()
    assert detect_parser(raw, openai_array) is OpenAIChatParser


def test_detect_openai_object(openai_chat):
    raw = openai_chat.read_text()
    assert detect_parser(raw, openai_chat) is OpenAIChatParser


def test_parse_claude_segments_and_turns(claude_jsonl):
    s = parse_file(claude_jsonl)
    assert s.source_format == "claude-code-jsonl"
    segs = _segments(s)
    assert Segment.SYSTEM in segs
    assert Segment.THINKING in segs
    assert Segment.TOOL_CALL in segs
    assert Segment.TOOL_RESULT in segs
    # summary line captured in meta, not as a message
    assert s.meta.get("summary")
    assert len(s.turns) == 4


def test_parse_claude_tool_ref_present(claude_jsonl):
    s = parse_file(claude_jsonl)
    calls = [m for m in s.messages if m.segment == Segment.TOOL_CALL]
    assert calls and all("Read" in (c.ref or "") for c in calls)
    assert any("config.py" in (c.ref or "") for c in calls)


def test_parse_codex(codex_session):
    s = parse_file(codex_session)
    assert s.source_format == "codex-session"
    segs = _segments(s)
    assert Segment.SYSTEM in segs
    assert Segment.TOOL_DEFINITIONS in segs
    assert Segment.THINKING in segs  # reasoning item
    assert Segment.TOOL_RESULT in segs
    assert s.meta.get("session_id") == "codex-abc123"


def test_parse_openai_object_has_tool_defs(openai_chat):
    s = parse_file(openai_chat)
    assert any(m.segment == Segment.TOOL_DEFINITIONS for m in s.messages)
    assert any(m.segment == Segment.TOOL_CALL for m in s.messages)
    assert any(m.segment == Segment.TOOL_RESULT for m in s.messages)


def test_parse_openai_array_roles(openai_array):
    s = parse_file(openai_array)
    assert any(m.segment == Segment.SYSTEM for m in s.messages)
    assert any(m.segment == Segment.USER for m in s.messages)
    assert any(m.segment == Segment.ASSISTANT for m in s.messages)


@pytest.mark.parametrize("wrapped", [False, True], ids=["array", "object"])
def test_detect_openai_developer_only(wrapped):
    def transcript(role):
        messages = [{"role": role, "content": "Answer concisely."}]
        return json.dumps({"messages": messages} if wrapped else messages)

    raw = transcript("developer")
    assert detect_parser(raw) is OpenAIChatParser
    assert OpenAIChatParser.sniff(raw) == OpenAIChatParser.sniff(transcript("system"))


@pytest.mark.parametrize("wrapped", [False, True], ids=["array", "object"])
def test_parse_openai_developer_role(wrapped):
    messages = [
        {"role": "developer", "content": "Answer concisely."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "developer", "content": "Use bullet points now."},
    ]
    raw = json.dumps({"messages": messages} if wrapped else messages)

    session = parse_text(raw)

    assert session.source_format == "openai-chat"
    assert [m.segment for m in session.messages] == [
        Segment.SYSTEM,
        Segment.USER,
        Segment.ASSISTANT,
        Segment.SYSTEM,
    ]
    assert [m.role for m in session.messages] == [m["role"] for m in messages]
    assert [m.text for m in session.messages] == [m["content"] for m in messages]
    assert [m.turn for m in session.messages] == [1, 1, 1, 2]


def test_force_format_override(openai_array):
    raw = openai_array.read_text()
    s = parse_text(raw, fmt="openai-chat")
    assert s.source_format == "openai-chat"


def test_unknown_format_raises():
    with pytest.raises(ParseError):
        parse_text("this is not a transcript at all")


def test_bad_json_line_raises():
    with pytest.raises(ParseError):
        ClaudeCodeParser().parse(
            '{"type":"user","message":{"role":"user","content":"hi"}}\n{bad json'
        )


def test_turns_are_monotonic(claude_jsonl, codex_session, openai_chat):
    for f in (claude_jsonl, codex_session, openai_chat):
        s = parse_file(f)
        turn_indices = [t.index for t in s.turns]
        assert turn_indices == sorted(turn_indices)
        assert turn_indices[0] == 1
