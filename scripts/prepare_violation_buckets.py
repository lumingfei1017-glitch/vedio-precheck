from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import openpyxl


ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT.parent
VIDEO_DIR = PROJECT_ROOT / "参考" / "审核建议"
EXCEL_GLOB = PROJECT_ROOT / "参考" / "审核建议_*.xlsx"
ASR_DIR = ROOT / "data" / "ref-eval" / "asr"
OUT_DIR = ROOT / "data" / "violation-buckets"

CURLY_QUOTE_RE = re.compile(r"“([^”]+)”")
STRAIGHT_QUOTE_RE = re.compile(r'"([^"\n]{1,})"')
SINGLE_QUOTE_RE = re.compile(r"‘([^’]+)’")
TIME_RE = re.compile(r"第?\s*(\d{1,3})(?:\.(\d{1,3}))?\s*秒")
CLOCK_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)")
NORMALIZE_RE = re.compile(r"[\s，。！？!?；;：:、,.·\"'“”‘’（）()\[\]【】…—\-]")
BODY_REQUIREMENTS = {
    "臀": ("臀",),
    "腰腹": ("腰", "腹"),
    "腹部": ("腹",),
    "肚子": ("肚",),
    "小腹": ("腹",),
    "大腿": ("腿",),
    "胸部": ("胸",),
    "后背": ("背",),
    "肩": ("肩",),
    "手臂": ("臂",),
}


def normalize(text: str) -> str:
    value = NORMALIZE_RE.sub("", text or "").lower()
    for source, target in (
        ("一百五十", "150"),
        ("一百五", "150"),
        ("一百二十", "120"),
        ("一百二", "120"),
        ("九十", "90"),
        ("十几", "10"),
        ("一二十", "10-20"),
        ("十来", "10"),
        ("三厘米", "3厘米"),
    ):
        value = value.replace(source, target)
    return value


def extract_quotes(suggestion: str) -> list[str]:
    found: list[str] = []
    for regex in (CURLY_QUOTE_RE, STRAIGHT_QUOTE_RE, SINGLE_QUOTE_RE):
        found.extend(regex.findall(suggestion or ""))
    result: list[str] = []
    for quote in found:
        quote = quote.strip()
        if quote and quote not in result:
            result.append(quote)
    return result


def extract_times_ms(text: str) -> list[int]:
    times: list[int] = []
    for match in TIME_RE.finditer(text or ""):
        seconds = int(match.group(1))
        fraction = match.group(2) or ""
        times.append(seconds * 1000 + int((fraction + "000")[:3]))
    for match in CLOCK_RE.finditer(text or ""):
        first = int(match.group(1))
        second = int(match.group(2))
        third = match.group(3)
        if third is None:
            times.append((first * 60 + second) * 1000)
        else:
            times.append((first * 3600 + second * 60 + int(third)) * 1000)
    return sorted(set(times))


def evidence_text(suggestion: str) -> str:
    start = max(suggestion.rfind("("), suggestion.rfind("（"))
    return suggestion[start + 1 :] if start >= 0 else suggestion


def without_quotes(text: str) -> str:
    for regex in (CURLY_QUOTE_RE, STRAIGHT_QUOTE_RE, SINGLE_QUOTE_RE):
        text = regex.sub("", text)
    return text


def source_kind(suggestion: str) -> str:
    has_screen = any(term in suggestion for term in ("视频画面", "画面", "镜头", "截图", "展示"))
    has_audio = any(term in suggestion for term in ("音频", "口播", "字幕", "文案", "话术", "声音"))
    if has_screen and has_audio:
        return "音画"
    if has_screen:
        return "画面"
    if has_audio:
        return "口播/字幕"
    return "未注明"


def clauses(text: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r"[，。！？!?；;、,：:\n…]+", text or "")
        if part.strip()
    ]
    return parts or [text.strip()]


def best_window(
    needle: str, segments: list[dict], preferred_ms: int | None = None
) -> dict | None:
    needle = normalize(needle)
    if not needle:
        return None

    best: dict | None = None
    for size in range(1, min(8, len(segments)) + 1):
        for index in range(0, len(segments) - size + 1):
            window = segments[index : index + size]
            norm_parts = [normalize(str(item.get("text") or "")) for item in window]
            haystack = "".join(norm_parts)
            if not haystack:
                continue

            if needle in haystack:
                score = 1.0
            else:
                matcher = SequenceMatcher(None, needle, haystack)
                longest = matcher.find_longest_match(0, len(needle), 0, len(haystack)).size
                score = max(matcher.ratio(), longest / len(needle))
            score = max(0.0, score - body_term_penalty(needle, haystack))
            score = max(0.0, score - leading_body_penalty(needle, norm_parts[0]))
            if leading_body_bonus(needle, norm_parts[0]):
                score = min(1.0, score + 0.08)

            start = int(window[0]["start_ms"])
            end = int(window[-1]["end_ms"])
            precise_start = start
            precise_end = end
            time_method = "句子级"
            if needle in haystack:
                position = haystack.find(needle)
                precise_start = offset_to_ms(window, norm_parts, position)
                precise_end = offset_to_ms(window, norm_parts, position + len(needle))
                time_method = "字符估算"
            candidate = {
                "score": score,
                "start_ms": start,
                "end_ms": end,
                "precise_start_ms": precise_start,
                "precise_end_ms": precise_end,
                "time_method": time_method,
                "matched_text": "".join(str(item.get("text") or "") for item in window),
            }

            if best is None:
                best = candidate
                continue

            score_gap = candidate["score"] - best["score"]
            if score_gap > 0.04:
                best = candidate
                continue
            if abs(score_gap) <= 0.04 and preferred_ms is not None:
                candidate_distance = abs((start + end) / 2 - preferred_ms)
                best_distance = abs((best["start_ms"] + best["end_ms"]) / 2 - preferred_ms)
                if candidate_distance < best_distance:
                    best = candidate
                continue
            if score_gap > 0:
                best = candidate
            elif candidate["score"] == best["score"] and end - start < best["end_ms"] - best["start_ms"]:
                best = candidate
    return best


def body_term_penalty(needle: str, haystack: str) -> float:
    penalty = 0.0
    for term, required_chars in BODY_REQUIREMENTS.items():
        if term in needle and not all(char in haystack for char in required_chars):
            penalty += 0.12
    return min(penalty, 0.24)


def leading_body_penalty(needle: str, first_text: str) -> float:
    for term, required_chars in BODY_REQUIREMENTS.items():
        if needle.startswith(term) and not all(char in first_text for char in required_chars):
            return 0.18
    return 0.0


def leading_body_bonus(needle: str, first_text: str) -> bool:
    return any(
        needle.startswith(term) and all(char in first_text for char in required_chars)
        for term, required_chars in BODY_REQUIREMENTS.items()
    )


def offset_to_ms(window: list[dict], norm_parts: list[str], offset: int) -> int:
    remaining = max(0, offset)
    for segment, norm_text in zip(window, norm_parts):
        length = len(norm_text)
        if remaining <= length:
            start = int(segment["start_ms"])
            end = int(segment["end_ms"])
            ratio = remaining / length if length else 0
            return int(start + (end - start) * ratio)
        remaining -= length
    return int(window[-1]["end_ms"])


def align_quote(
    quote: str, segments: list[dict], preferred_ms: int | None = None
) -> dict | None:
    matched_clauses: list[dict] = []
    total_length = 0
    matched_length = 0
    clause_details: list[dict] = []

    for clause in clauses(quote):
        needle = normalize(clause)
        if not needle:
            continue
        total_length += len(needle)
        match = best_window(clause, segments, preferred_ms)
        threshold = 0.82 if len(needle) < 6 else 0.72
        accepted = match is not None and match["score"] >= threshold
        if accepted:
            matched_length += len(needle)
            matched_clauses.append(match)
        clause_details.append(
            {
                "clause": clause,
                "score": round(match["score"], 3) if match else 0,
                "accepted": accepted,
            }
        )

    if not matched_clauses or total_length == 0:
        return {
            "start_ms": None,
            "end_ms": None,
            "duration_ms": None,
            "match_score": 0,
            "match_coverage": 0,
            "matched_text": "",
            "matched_clauses": 0,
            "total_clauses": len(clause_details),
            "clause_details": clause_details,
        }

    start = min(item.get("precise_start_ms", item["start_ms"]) for item in matched_clauses)
    end = max(item.get("precise_end_ms", item["end_ms"]) for item in matched_clauses)
    time_method = (
        "字符估算"
        if all(item.get("time_method") == "字符估算" for item in matched_clauses)
        else "句子级"
    )
    coverage = matched_length / total_length
    matched_segments = [
        item
        for item in segments
        if int(item["end_ms"]) > start and int(item["start_ms"]) < end
    ]
    return {
        "start_ms": start,
        "end_ms": end,
        "duration_ms": end - start,
        "match_score": round(sum(item["score"] for item in matched_clauses) / len(matched_clauses), 3),
        "match_coverage": round(coverage, 3),
        "time_method": time_method,
        "matched_text": "".join(str(item.get("text") or "") for item in matched_segments),
        "matched_clauses": len(matched_clauses),
        "total_clauses": len(clause_details),
        "clause_details": clause_details,
    }


def bucket_for(duration_ms: int) -> str:
    if duration_ms <= 5000:
        return "5秒以内"
    if duration_ms <= 15000:
        return "5-15秒"
    return "15秒以上"


def main() -> int:
    excel_candidates = [
        path for path in EXCEL_GLOB.parent.glob(EXCEL_GLOB.name) if not path.name.startswith("~$")
    ]
    if not excel_candidates:
        raise FileNotFoundError(f"找不到审核建议 Excel: {EXCEL_GLOB}")
    excel_path = max(excel_candidates, key=lambda path: path.stat().st_mtime)

    workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    sheet = workbook["审核建议明细"]
    rows = list(sheet.iter_rows(min_row=2, values_only=True))

    local_video_ids = {path.stem for path in VIDEO_DIR.glob("*.mp4")}
    asr: dict[str, list[dict]] = {}
    for path in ASR_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("segments"):
            asr[path.stem] = payload["segments"]

    points: list[dict] = []
    unmatched: list[dict] = []
    local_rows = 0
    quote_count = 0

    for row_number, row in enumerate(rows, start=2):
        video_id = str(row[6] or "").strip()
        material_id = str(row[5] or "").strip()
        category = str(row[2] or "").strip()
        suggestion = str(row[9] or "").strip()
        quotes = extract_quotes(suggestion)
        quote_count += len(quotes)

        if video_id not in local_video_ids:
            unmatched.append(
                {
                    "source_row": row_number,
                    "material_id": material_id,
                    "video_id": video_id,
                    "category": category,
                    "quote": "；".join(quotes),
                    "reason": "Excel 视频 ID 无本地视频",
                    "suggestion": suggestion,
                }
            )
            continue

        local_rows += 1
        if not quotes:
            unmatched.append(
                {
                    "source_row": row_number,
                    "material_id": material_id,
                    "video_id": video_id,
                    "category": category,
                    "quote": "",
                    "reason": "审核建议未提供引文",
                    "source_kind": source_kind(suggestion),
                    "suggestion": suggestion,
                }
            )
            continue

        explicit_times = extract_times_ms(without_quotes(evidence_text(suggestion)))
        if video_id not in asr:
            for quote in quotes:
                unmatched.append(
                    {
                        "source_row": row_number,
                        "material_id": material_id,
                        "video_id": video_id,
                        "category": category,
                        "quote": quote,
                        "reason": "没有 ASR 数据",
                        "source_kind": source_kind(suggestion),
                        "suggestion": suggestion,
                    }
                )
            continue

        for quote in quotes:
            preferred_ms = explicit_times[0] if len(explicit_times) == 1 else None
            aligned = align_quote(quote, asr[video_id], preferred_ms)
            if (
                aligned is None
                or aligned["start_ms"] is None
                or aligned["match_coverage"] < 0.55
                or aligned["matched_clauses"] == 0
            ):
                reason = (
                    "画面类违规，ASR 不可定位"
                    if source_kind(suggestion) == "画面"
                    else "ASR 中未找到完整对应片段"
                )
                unmatched.append(
                    {
                        "source_row": row_number,
                        "material_id": material_id,
                        "video_id": video_id,
                        "category": category,
                        "quote": quote,
                        "reason": reason,
                        "source_kind": source_kind(suggestion),
                        "explicit_time_ms": preferred_ms,
                        "match_score": aligned.get("match_score") if aligned else 0,
                        "match_coverage": aligned.get("match_coverage") if aligned else 0,
                        "matched_text": aligned.get("matched_text") if aligned else "",
                        "clause_details": aligned.get("clause_details") if aligned else [],
                        "suggestion": suggestion,
                    }
                )
                continue

            status = "完整匹配" if aligned["match_coverage"] >= 0.88 else "部分匹配"
            points.append(
                {
                    "source_row": row_number,
                    "material_id": material_id,
                    "video_id": video_id,
                    "category": category,
                    "quote": quote,
                    "start_ms": aligned["start_ms"],
                    "end_ms": aligned["end_ms"],
                    "duration_ms": aligned["duration_ms"],
                    "bucket": bucket_for(aligned["duration_ms"]),
                    "match_status": status,
                    "match_score": aligned["match_score"],
                    "match_coverage": aligned["match_coverage"],
                    "time_method": aligned["time_method"],
                    "matched_clauses": aligned["matched_clauses"],
                    "total_clauses": aligned["total_clauses"],
                    "matched_text": aligned["matched_text"],
                    "source_kind": source_kind(suggestion),
                    "explicit_time_ms": preferred_ms,
                    "suggestion": suggestion,
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "points.json").write_text(
        json.dumps(points, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "unmatched.json").write_text(
        json.dumps(unmatched, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bucket_counts = {bucket: 0 for bucket in ("5秒以内", "5-15秒", "15秒以上")}
    for point in points:
        bucket_counts[point["bucket"]] += 1
    status_counts = Counter(point["match_status"] for point in points)
    reason_counts = Counter(item["reason"] for item in unmatched)
    summary = {
        "excel": str(excel_path),
        "excel_rows": len(rows),
        "local_video_rows": local_rows,
        "local_video_count": len(local_video_ids),
        "asr_video_count": len(asr),
        "quote_count": quote_count,
        "located_points": len(points),
        "bucket_counts": bucket_counts,
        "match_status_counts": dict(status_counts),
        "unmatched_count": len(unmatched),
        "unmatched_reasons": dict(reason_counts),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
