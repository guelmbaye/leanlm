"""Structure detection and segmentation (DDPB stages 4-6).

Structure comes before AI (DP-03). A heading tells us what a paragraph is
*about*, which later lets retrieval boost a passage whose section title matches
the question -- a cheap signal that costs no tokens and no model call.
"""
from __future__ import annotations

import re
from pathlib import Path

from ...contracts.dto import SemanticUnit, StructuredDocument, UnitKind
from ...shared.hashing import sha256_text
from ...shared.text import DEFAULT_TOKEN_COUNTER, TokenCounter, sentences, words
from .models import Block
from .policies import SegmentationPolicy

_ATX = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")
_NUMBERED = re.compile(r"^((?:\d+\.){1,4}|\d+\))\s+(\S.{0,110})$")
_BULLET = re.compile(r"^\s{0,6}([-*\u2022\u2023\u25cf]|\d+[.)])\s+\S")
_TABLE = re.compile(r"^\s*\|.*\|\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

# Language hint: a handful of high-frequency function words is enough to tell
# FR from EN, and the hint only ever influences reporting, never the pipeline.
_FR_MARKERS = frozenset("les des une dans pour que qui est sont avec sur cette aux".split())
_EN_MARKERS = frozenset("the and for that with this are was from have been will".split())


class StructureAnalyzer:
    """Detects headings, lists, tables and code blocks in normalized text."""

    def analyze(self, text: str) -> list[Block]:
        blocks: list[Block] = []
        lines = text.split("\n")
        offsets: list[int] = []
        cursor = 0
        for line in lines:
            offsets.append(cursor)
            cursor += len(line) + 1

        buffer: list[str] = []
        buffer_start = 0
        buffer_kind = UnitKind.PARAGRAPH
        in_fence = False

        def flush(end: int) -> None:
            nonlocal buffer, buffer_kind
            if buffer:
                body = "\n".join(buffer).strip()
                if body:
                    blocks.append(Block(buffer_kind, body, buffer_start, end))
                buffer = []
                buffer_kind = UnitKind.PARAGRAPH

        index = 0
        while index < len(lines):
            line = lines[index]
            start = offsets[index]
            end = start + len(line)

            if _FENCE.match(line):
                if in_fence:
                    buffer.append(line)
                    flush(end)
                    in_fence = False
                else:
                    flush(start)
                    in_fence = True
                    buffer_start = start
                    buffer_kind = UnitKind.CODE
                    buffer.append(line)
                index += 1
                continue
            if in_fence:
                buffer.append(line)
                index += 1
                continue

            if not line.strip():
                flush(start)
                index += 1
                continue

            atx = _ATX.match(line)
            if atx:
                flush(start)
                blocks.append(Block(UnitKind.HEADING, atx.group(2), start, end,
                                    len(atx.group(1))))
                index += 1
                continue

            nxt = lines[index + 1] if index + 1 < len(lines) else ""
            if _SETEXT.match(nxt) and line.strip() and len(line.strip()) <= 120:
                flush(start)
                level = 1 if nxt.strip().startswith("=") else 2
                blocks.append(Block(UnitKind.HEADING, line.strip(), start, end, level))
                index += 2
                continue

            numbered = _NUMBERED.match(line.strip())
            if numbered and self._looks_like_heading(line, nxt):
                flush(start)
                level = min(6, line.strip().split()[0].count(".") + 1)
                blocks.append(Block(UnitKind.HEADING, line.strip(), start, end, level))
                index += 1
                continue

            if self._is_caps_heading(line):
                flush(start)
                blocks.append(Block(UnitKind.HEADING, line.strip(), start, end, 2))
                index += 1
                continue

            kind = UnitKind.PARAGRAPH
            if _TABLE.match(line):
                kind = UnitKind.TABLE
            elif _BULLET.match(line):
                kind = UnitKind.LIST

            if buffer and kind is not buffer_kind:
                flush(start)
            if not buffer:
                buffer_start = start
                buffer_kind = kind
            buffer.append(line)
            index += 1

        flush(offsets[-1] + len(lines[-1]) if lines else 0)
        return blocks

    @staticmethod
    def _looks_like_heading(line: str, following: str) -> bool:
        body = line.strip()
        if len(body) > 110 or body.endswith((".", ";", ",")):
            return False
        return bool(following.strip()) or len(body.split()) <= 12

    @staticmethod
    def _is_caps_heading(line: str) -> bool:
        body = line.strip()
        if not (3 <= len(body) <= 90) or body.endswith("."):
            return False
        letters = [c for c in body if c.isalpha()]
        if len(letters) < 3:
            return False
        return all(c.isupper() for c in letters) and len(body.split()) <= 10


class Segmenter:
    """Turns blocks into SemanticUnits carrying their section path."""

    def __init__(self, policy: SegmentationPolicy | None = None,
                 counter: TokenCounter | None = None) -> None:
        self.policy = policy or SegmentationPolicy()
        self.counter = counter or DEFAULT_TOKEN_COUNTER

    def segment(self, blocks: list[Block], *, document_id: str,
                document_name: str) -> list[SemanticUnit]:
        units: list[SemanticUnit] = []
        section_path: list[str] = []
        pending: list[Block] = []
        order = 0

        def emit(collected: list[Block]) -> None:
            nonlocal order
            if not collected:
                return
            body = "\n\n".join(b.text for b in collected).strip()
            if not body:
                collected.clear()
                return
            tokens = self.counter.count(body)
            if tokens > self.policy.max_tokens:
                for piece in self._split_oversized(body):
                    order = self._append(units, piece, collected[0], section_path,
                                         document_id, document_name, order)
            else:
                order = self._append(units, body, collected[0], section_path,
                                     document_id, document_name, order)
            collected.clear()

        for block in blocks:
            if block.kind is UnitKind.HEADING:
                emit(pending)
                level = max(1, block.heading_level)
                del section_path[level - 1:]
                section_path.append(block.text.strip())
                if self.policy.keep_headings_as_units:
                    order = self._append(units, block.text.strip(), block, section_path,
                                         document_id, document_name, order,
                                         kind=UnitKind.HEADING)
                continue
            pending.append(block)
            merged_tokens = self.counter.count("\n\n".join(b.text for b in pending))
            if merged_tokens >= self.policy.merge_below_tokens:
                emit(pending)
            if len(units) >= self.policy.max_units_per_document:
                break
        emit(pending)
        return units

    def _append(self, units: list[SemanticUnit], text: str, block: Block,
                section_path: list[str], document_id: str, document_name: str,
                order: int, *, kind: UnitKind | None = None) -> int:
        tokens = self.counter.count(text)
        foldable = (
            kind is not UnitKind.HEADING
            and tokens < self.policy.min_tokens
            and units
            and units[-1].kind is not UnitKind.HEADING
            and units[-1].section_path == tuple(section_path)
        )
        if foldable:
            # Too small to stand alone: fold it into the previous unit of the
            # same section rather than paying a retrieval slot for a fragment.
            # Never fold across a heading -- that would corrupt the section map.
            previous = units[-1]
            merged_text = f"{previous.text}\n{text}"
            units[-1] = self._make_unit(
                merged_text, previous.kind, previous.section_path, document_id,
                document_name, previous.order, previous.char_start, block.char_end,
                previous.heading_level,
            )
            return order
        unit = self._make_unit(
            text, kind or block.kind, tuple(section_path), document_id, document_name,
            order, block.char_start, block.char_end, block.heading_level,
        )
        units.append(unit)
        return order + 1

    def _make_unit(self, text: str, kind: UnitKind, section_path: tuple[str, ...],
                   document_id: str, document_name: str, order: int, start: int,
                   end: int, heading_level: int) -> SemanticUnit:
        unit_id = f"{document_id[:12]}#{order:04d}"
        unit = SemanticUnit(
            meta=SemanticUnit.build_meta(
                identity=(document_id, str(order), text[:120]),
                parent_id=document_id, source=document_name, stage="segmentation",
            ),
            unit_id=unit_id,
            document_id=document_id,
            document_name=document_name,
            kind=kind,
            text=text,
            section_path=tuple(section_path),
            order=order,
            char_start=start,
            char_end=end,
            token_estimate=self.counter.count(text),
            heading_level=heading_level,
        )
        return unit.sealed()  # type: ignore[return-value]

    def _split_oversized(self, text: str) -> list[str]:
        pieces: list[str] = []
        current: list[str] = []
        used = 0
        for sentence in sentences(text):
            cost = self.counter.count(sentence)
            if current and used + cost > self.policy.target_tokens:
                pieces.append(" ".join(current))
                current, used = [], 0
            current.append(sentence)
            used += cost
        if current:
            pieces.append(" ".join(current))
        return pieces or [text]


def detect_language(text: str) -> str:
    sample = words(text[:4000])
    if not sample:
        return "und"
    fr = sum(1 for w in sample if w in _FR_MARKERS)
    en = sum(1 for w in sample if w in _EN_MARKERS)
    if fr == en:
        return "und"
    return "fr" if fr > en else "en"


def build_structured_document(normalized_text: str, *, filename: str,
                              descriptor_id: str,
                              policy: SegmentationPolicy | None = None,
                              counter: TokenCounter | None = None) -> StructuredDocument:
    counter = counter or DEFAULT_TOKEN_COUNTER
    blocks = StructureAnalyzer().analyze(normalized_text)
    document_id = sha256_text(f"{descriptor_id}:{filename}")[:24]
    units = Segmenter(policy, counter).segment(
        blocks, document_id=document_id, document_name=filename
    )
    headings = tuple(b.text for b in blocks if b.kind is UnitKind.HEADING)
    title = headings[0] if headings else Path(filename).stem.replace("_", " ")
    document = StructuredDocument(
        meta=StructuredDocument.build_meta(
            identity=(document_id, str(len(units))), parent_id=descriptor_id,
            source=filename, stage="structure_detection",
        ),
        document_id=document_id,
        filename=filename,
        title=title,
        units=tuple(units),
        section_titles=headings,
        token_estimate=sum(u.token_estimate for u in units),
        language_hint=detect_language(normalized_text),
    )
    return document.sealed()  # type: ignore[return-value]
