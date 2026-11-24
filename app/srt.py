from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

TIMESTAMP_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})")


@dataclass(frozen=True)
class Segment:
    index: int
    start_ms: int
    end_ms: int
    text: str


def parse_timestamp(value: str) -> int:
    match = TIMESTAMP_RE.search(value)
    if not match:
        raise ValueError(f"invalid timestamp: {value}")
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def parse_srt(content: str) -> list[Segment]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = re.split(r"\n{2,}", content.strip())
    segments: list[Segment] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        timestamp_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timestamp_index is None:
            continue
        left, _, right = lines[timestamp_index].partition("-->")
        try:
            start_ms = parse_timestamp(left)
            end_ms = parse_timestamp(right)
        except ValueError:
            continue
        text = " ".join(lines[timestamp_index + 1 :]).strip()
        if not text:
            continue
        segments.append(Segment(index=len(segments) + 1, start_ms=start_ms, end_ms=end_ms, text=text))

    return segments


def normalize(text: str) -> str:
    return re.sub(r"[\s，。！？!?；;：:、,.·\"'“”‘’（）()\[\]【】]", "", text)


def align_quote(quote: str, segments: list[Segment]) -> tuple[int | None, int | None, float]:
    normalized_quote = normalize(quote)
    if not normalized_quote or not segments:
        return None, None, 0.0

    for segment in segments:
        normalized_segment = normalize(segment.text)
        if not normalized_segment:
            continue
        if normalized_quote in normalized_segment:
            return segment.start_ms, segment.end_ms, 0.95

    # A quote can span several subtitle lines; prefer the smallest exact window
    # before accepting a single fragment that happens to be contained in it.
    for size in (2, 3):
        for index in range(0, len(segments) - size + 1):
            window = segments[index : index + size]
            combined = normalize("".join(segment.text for segment in window))
            if normalized_quote in combined:
                return window[0].start_ms, window[-1].end_ms, 0.95

    for segment in segments:
        normalized_segment = normalize(segment.text)
        if len(normalized_segment) >= 4 and normalized_segment in normalized_quote:
            return segment.start_ms, segment.end_ms, 0.85

    best_ratio = 0.0
    best_segment: Segment | None = None
    for segment in segments:
        ratio = difflib.SequenceMatcher(None, normalized_quote, normalize(segment.text)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_segment = segment

    if best_segment is not None and best_ratio >= 0.45:
        return best_segment.start_ms, best_segment.end_ms, round(best_ratio, 2)

    return None, None, 0.0


def text_from_segments(segments: list[Segment]) -> str:
    return "\n".join(segment.text for segment in segments)


def format_timestamp(ms: int) -> str:
    value = max(0, int(ms))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def render_srt(segments: list[Segment]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        end_ms = max(segment.end_ms, segment.start_ms + 200)
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(segment.start_ms)} --> {format_timestamp(end_ms)}\n"
            f"{segment.text.strip()}\n"
        )
    return "\n".join(blocks)


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")
