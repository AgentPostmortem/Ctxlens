"""Turn the parsed :class:`Session` into a token profile.

The profile is the central analysis object every reporter consumes. It holds
per-segment totals, per-turn cumulative growth (context high-water mark), and
the single biggest context consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ctxlens.models import Segment, Session
from ctxlens.tokenizers.base import Tokenizer


@dataclass
class SegmentStat:
    segment: Segment
    tokens: int
    count: int

    def pct_of(self, total: int) -> float:
        return (self.tokens / total * 100.0) if total else 0.0


@dataclass
class TurnStat:
    index: int
    tokens: int
    cumulative: int
    by_segment: dict[Segment, int] = field(default_factory=dict)


@dataclass
class Consumer:
    """A single message that is a large slice of context."""

    label: str
    segment: Segment
    tokens: int
    turn: int
    ref: str | None = None


@dataclass
class Profile:
    session: Session
    tokenizer_name: str
    total_tokens: int
    segment_stats: list[SegmentStat]
    turn_stats: list[TurnStat]
    top_consumers: list[Consumer]

    @property
    def high_water_mark(self) -> int:
        """Largest cumulative context size reached (equals total for an
        append-only transcript, but computed independently)."""
        return max((t.cumulative for t in self.turn_stats), default=0)

    def segment_map(self) -> dict[Segment, SegmentStat]:
        return {s.segment: s for s in self.segment_stats}


def build_profile(
    session: Session, tokenizer: Tokenizer, *, top_n: int = 10
) -> Profile:
    """Count tokens for every message and aggregate them into a Profile."""
    for m in session.messages:
        m.tokens = tokenizer.count(m.text)
    session._tokens_counted = True

    # per-segment totals
    seg_tokens: dict[Segment, int] = {}
    seg_counts: dict[Segment, int] = {}
    for m in session.messages:
        seg_tokens[m.segment] = seg_tokens.get(m.segment, 0) + m.tokens
        seg_counts[m.segment] = seg_counts.get(m.segment, 0) + 1
    segment_stats = [
        SegmentStat(seg, seg_tokens[seg], seg_counts[seg])
        for seg in sorted(seg_tokens, key=lambda s: seg_tokens[s], reverse=True)
    ]

    # per-turn stats with cumulative growth
    turn_stats: list[TurnStat] = []
    cumulative = 0
    for turn in session.turns:
        by_seg: dict[Segment, int] = {}
        for m in turn.messages:
            by_seg[m.segment] = by_seg.get(m.segment, 0) + m.tokens
        cumulative += turn.tokens
        turn_stats.append(
            TurnStat(index=turn.index, tokens=turn.tokens, cumulative=cumulative, by_segment=by_seg)
        )

    # biggest single consumers
    ranked = sorted(session.messages, key=lambda m: m.tokens, reverse=True)
    top_consumers = [
        Consumer(
            label=_consumer_label(m),
            segment=m.segment,
            tokens=m.tokens,
            turn=m.turn,
            ref=m.ref,
        )
        for m in ranked[:top_n]
        if m.tokens > 0
    ]

    return Profile(
        session=session,
        tokenizer_name=tokenizer.name,
        total_tokens=session.total_tokens,
        segment_stats=segment_stats,
        turn_stats=turn_stats,
        top_consumers=top_consumers,
    )


def _consumer_label(m) -> str:
    if m.tool_name:
        return f"{m.segment.label}: {m.tool_name}"
    if m.ref:
        return f"{m.segment.label}: {m.ref}"
    snippet = " ".join(m.text.split())[:48]
    return f"{m.segment.label}: {snippet}" if snippet else m.segment.label
