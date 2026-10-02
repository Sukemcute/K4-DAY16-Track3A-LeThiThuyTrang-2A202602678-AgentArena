"""Evidence shared by the report layers, never an answer generator.

Only successful tool payloads that survived into an observation establish
source ownership. Corpus lines validate those payloads; unseen corpus text
never becomes a claim. All caches belong to AgentContext, not middleware.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 500
MAX_CLAIMS = 10
MAX_PER_DOC = 4
MAX_INPUT_CLAIMS = 1000
_SCOPES_KEY = "evidence.scopes"


def json_payload(text):
    """Decode bounded external JSON, rejecting malformed/deep wire data.

    This narrow decoding boundary handles data errors, not hook bugs.
    """
    if not isinstance(text, str) or len(text) > 100_000:
        return None
    if not text.lstrip().startswith(("[", "{")):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return None


def remember_result(ctx, name, args, result):
    """Record returned source scopes; the index later checks visibility.

    Inner layers may see an unfiltered payload. It is NOT trusted unless
    the same payload actually reached ctx.observations after outer guards.
    """
    if not getattr(result, "ok", False) or not isinstance(result.content, str):
        return
    scopes = ctx.state.setdefault(_SCOPES_KEY, {})
    if name == "fetch_doc" and isinstance(args, dict):
        pairs = [(args.get("doc_id"), result.content)]
    elif name == "search":
        payload = json_payload(result.content)
        pairs = [(row.get("doc_id"), row.get("snippet"))
                 for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
    else:
        return
    for doc_id, content in pairs:
        if not isinstance(doc_id, str) or not isinstance(content, str) or not content:
            continue
        bucket = scopes.setdefault(doc_id, [])
        if content not in bucket:
            bucket.append(content)


def observations(ctx):
    raw = getattr(ctx, "observations", None)
    if not isinstance(raw, list):
        raw = [getattr(ctx, "observed_text", "")]
    visible = [text for text in raw if isinstance(text, str) and text]
    # Search is JSON-encoded. Decode snippets as data without treating JSON
    # syntax, doc IDs or titles as quotable document evidence.
    for text in list(visible):
        payload = json_payload(text)
        if isinstance(payload, list):
            visible.extend(row["snippet"] for row in payload
                           if isinstance(row, dict) and isinstance(row.get("snippet"), str))
    return visible


@dataclass(frozen=True)
class Source:
    doc_id: str
    lines: tuple[str, ...]
    scopes: tuple[str, ...]

    @property
    def subject(self):
        for scope in self.scopes:
            for line in scope.splitlines():
                if line.casefold().startswith("chủ đề:"):
                    return content_words(line.partition(":")[2])
        return set()

    @property
    def superseded(self):
        return any(re.search(r"đã (?:bị |được )?thay thế|đã hết hiệu lực|superseded|deprecated",
                             scope, re.IGNORECASE) for scope in self.scopes)

    def supports(self, text):
        return (isinstance(text, str) and len(" ".join(text.split())) >= MIN_QUOTE_CHARS
                and any(text in line for line in self.lines)
                and any(text in scope for scope in self.scopes))


class EvidenceIndex:
    """A deterministic snapshot of actually visible, source-scoped quotes."""

    def __init__(self, ctx):
        visible = observations(ctx)
        recorded = ctx.state.get(_SCOPES_KEY, {})
        corpus = getattr(ctx, "corpus", None)
        self.sources = []
        for doc in sorted(getattr(corpus, "docs", []), key=lambda d: d.doc_id):
            if not doc.body:
                continue
            scopes = [part for part in recorded.get(doc.doc_id, [])
                      if any(part in observation for observation in visible)]
            # Supports direct hook use and other stacks that do not record
            # scopes, but never infer a source from an isolated shared line.
            if _SCOPES_KEY not in ctx.state and any(doc.body in observation for observation in visible):
                scopes.append(doc.body)
            if scopes:
                self.sources.append(Source(doc.doc_id, tuple(doc.body.splitlines()), tuple(scopes)))

    def matches(self, text):
        return [source for source in self.sources if source.supports(text)]

    def source_for(self, text, preferred=None):
        matches = self.matches(text)
        return next((s for s in matches if s.doc_id == preferred), matches[0] if matches else None)

    def comparable(self, left, right):
        a = self.source_for(left.get("text"), left.get("doc_id"))
        b = self.source_for(right.get("text"), right.get("doc_id"))
        if not a or not b or a.doc_id == b.doc_id or a.superseded or b.superseded:
            return False
        # Numbers about different subjects are separate facts, even when
        # the documents use the same company-wide report/policy template.
        return not (a.subject and b.subject and a.subject != b.subject)


def citations(claims):
    return sorted({c["doc_id"] for c in claims
                   if isinstance(c, dict) and isinstance(c.get("doc_id"), str) and c["doc_id"]})


_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_STOP = frozenset("và là của các có cho trong được một những với này đó theo phải khi để thì từ hoặc the a an and of to in is are".split())


def content_words(text):
    return set(_WORD.findall(text.casefold())) - _STOP


def conflict(left, right, connector=""):
    """Conservative lexical conflict signal, not an oracle or topic table.

    Different numbers alone (e.g. delivery days vs refund hours) are not
    enough. Require shared subject words and the same numeric unit, or an
    explicit adversative with shared subject words.
    """
    shared = content_words(left) & content_words(right)
    if len(shared) < 2:
        return False
    values = re.compile(r"(\d+(?:[.,]\d+)?)\s*([^\W\d_]+|%)", re.UNICODE)
    units = {"ngày", "giờ", "tuần", "tháng", "days", "day", "hours", "hour", "weeks", "week", "%", "vnd", "đồng"}
    a = [m for m in values.finditer(left.casefold()) if m[2] in units]
    b = [m for m in values.finditer(right.casefold()) if m[2] in units]
    numeric = False
    for x in a:
        for y in b:
            if x[2] != y[2] or x[1] == y[1]:
                continue
            prefix_a = _WORD.findall(left[:x.start()].casefold())[-5:]
            prefix_b = _WORD.findall(right[:y.start()].casefold())[-5:]
            # Shared policy constraint and subject, not different counts in
            # unrelated reports using the same boilerplate wording.
            suffix = 0
            for p, q in zip(reversed(prefix_a), reversed(prefix_b)):
                if p != q:
                    break
                suffix += 1
            numeric |= suffix >= 3
    negative = any(word in connector.casefold() for word in ("nhưng", "trong khi", "but", "whereas"))
    return numeric or negative
