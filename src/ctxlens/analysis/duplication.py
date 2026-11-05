"""Duplicate-content detection.

Two independent signals:

* **ref duplication** - the same tool acted on the same target more than once
  (e.g. the same file re-read repeatedly). Keyed on :attr:`Message.ref`.
* **body duplication** - byte-identical message bodies appearing more than
  once, regardless of ref (e.g. the same large error pasted back several
  times).

Wasted tokens are counted as every repeat *after the first* occurrence, since
the first occurrence is legitimately needed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ctxlens.models import Message, Segment, Session


@dataclass
class DuplicateGroup:
    key: str
    kind: str  # "ref" or "body"
    segment: Segment
    occurrences: int
    tokens_each: int
    wasted_tokens: int
    turns: list[int] = field(default_factory=list)
    sample: str = ""


def _body_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def find_duplicates(session: Session, *, min_tokens: int = 20) -> list[DuplicateGroup]:
    """Return duplicate groups sorted by wasted tokens, descending.

    ``min_tokens`` filters out trivially small bodies (e.g. an empty tool
    result) that are not worth flagging.

    Nonempty sessions with no token counts must be passed to ``build_profile``
    first; otherwise a ``ValueError`` prevents a misleading empty result.
    """
    if (
        not session._tokens_counted
        and all(m.tokens == 0 for m in session.messages)
        and any(m.text.strip() for m in session.messages)
    ):
        raise ValueError(
            "Session has no token counts; call build_profile(session, tokenizer) "
            "before analyzing waste or duplicates."
        )
    groups: dict[tuple[str, str], list[Message]] = {}

    for m in session.messages:
        if m.tokens < min_tokens:
            continue
        # ref-based grouping only makes sense for tool activity, and only
        # counts as duplication when the *body* is identical too (a re-read
        # that returns changed content is "stale", handled by the waste report)
        if m.ref and m.segment in {Segment.TOOL_CALL, Segment.TOOL_RESULT}:
            groups.setdefault(("ref", f"{m.ref}\x00{_body_hash(m.text)}"), []).append(m)
        # body-based grouping applies to any repeated content
        groups.setdefault(("body", _body_hash(m.text)), []).append(m)

    results: list[DuplicateGroup] = []
    seen_bodies: set[str] = set()

    for (kind, _key), msgs in groups.items():
        if len(msgs) < 2:
            continue
        # avoid double-counting: if a ref group and a body group cover the same
        # messages, keep the ref group (more specific) and skip the body one
        if kind == "ref":
            seen_bodies.add(_body_hash(msgs[0].text))

    for (kind, key), msgs in groups.items():
        if len(msgs) < 2:
            continue
        body_sig = _body_hash(msgs[0].text)
        if kind == "body" and body_sig in seen_bodies:
            continue
        tokens_each = max(m.tokens for m in msgs)
        wasted = tokens_each * (len(msgs) - 1)
        display_key = key.split("\x00", 1)[0] if kind == "ref" else msgs[0].ref or key[:12]
        results.append(
            DuplicateGroup(
                key=display_key,
                kind=kind,
                segment=msgs[0].segment,
                occurrences=len(msgs),
                tokens_each=tokens_each,
                wasted_tokens=wasted,
                turns=sorted({m.turn for m in msgs}),
                sample=" ".join(msgs[0].text.split())[:80],
            )
        )

    results.sort(key=lambda g: g.wasted_tokens, reverse=True)
    return results
