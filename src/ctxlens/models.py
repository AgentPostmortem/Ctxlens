"""Core data model shared across parsers, analysis, and reporters.

A parsed transcript is a list of :class:`Message` objects grouped into
:class:`Turn` objects. Every message carries a :class:`Segment` role so the
analysis layer can attribute tokens to a category (system prompt, tool
definitions, tool results, user, assistant, thinking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Segment(StrEnum):
    """Category a chunk of context belongs to.

    The value is the human-readable label used in reports.
    """

    SYSTEM = "system"
    TOOL_DEFINITIONS = "tool_definitions"
    USER = "user"
    ASSISTANT = "assistant"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


@dataclass
class Message:
    """A single unit of context within a session.

    ``ref`` is an optional identifier used to detect duplication, e.g. a file
    path that was read, or a tool name. ``turn`` is the 1-based turn index the
    message belongs to.
    """

    segment: Segment
    text: str
    turn: int
    role: str = ""
    tool_name: str | None = None
    ref: str | None = None
    tokens: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Turn:
    """A logical turn: everything added to context up to and including one
    assistant response."""

    index: int
    messages: list[Message] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(m.tokens for m in self.messages)


@dataclass
class Session:
    """A fully parsed agent session."""

    source_format: str
    messages: list[Message] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    # A tokenizer may legitimately count nonempty text as zero tokens.
    _tokens_counted: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def turns(self) -> list[Turn]:
        by_index: dict[int, Turn] = {}
        for m in self.messages:
            by_index.setdefault(m.turn, Turn(index=m.turn)).messages.append(m)
        return [by_index[i] for i in sorted(by_index)]

    @property
    def total_tokens(self) -> int:
        return sum(m.tokens for m in self.messages)
