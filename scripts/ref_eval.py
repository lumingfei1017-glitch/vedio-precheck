from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.asr import AsrError, transcribe_video  # noqa: E402
from app.review import review_transcript  # noqa: E402
from app.srt import Segment, text_from_segments  # noqa: E402

REF_DIR = ROOT.parent / "参考"
VIDEO_DIR = REF_DIR / "审核建议"
EXCEL_PATH = REF_DIR / "审核建议_HY-北极绒官方旗舰店_红包-乘方04-21 18.25_20261119.xlsx"
OUT_DIR = ROOT / "data" / "ref-eval"
ASR_DIR = OUT_DIR / "asr"
RESULT_DIR = OUT_DIR / "results"
QUOTE_RE = re.compile(r"“([^”]+)”")
NORMALIZE_RE = re.compile(r"[\s，。！？!?；;：:、,.·\"'“”‘’（）()\[\]【】]")
NUMBER_REPLACEMENTS = (
    ("一百五十", "150"),
    ("一百五", "150"),
    ("一百二十", "120"),
    ("一百二", "120"),
    ("九十", "90"),
)


@dataclass(frozen=True)
class ExcelRow:
    video_id: str
    category: str
    suggestion: str


def normalize(text: str) -> str:
    value = NORMALIZE_RE.sub("", text or "")
    for source, target in NUMBER_REPLACEMENTS:
        value = value.replace(source, target)
    return value


def load_excel() -> dict[str, list[ExcelRow]]:
    workbook = openpyxl.load_workbook(EXCEL_PATH, read_only=True, data_only=True)
    sheet = workbook["审核建议明细"]
    by_id: dict[str, list[ExcelRow]] = defaultdict(list)
    for row in sheet.iter_rows(min_row=2, values_only=True):
        video_id = str(row[6] or "").strip()
        if not video_id:
            continue
        by_id[video_id].append(
            ExcelRow(
                video_id=video_id,
                category=str(row[2] or "").strip(),
                suggestion=str(row[9] or "").strip(),
            )
        )
    return dict(by_id)


def _candidate_hit(candidate: str, findings: list[dict[str, Any]]) -> bool:
    for finding in findings:
        normalized_finding = normalize(finding["quote"])
        if not normalized_finding:
            continue
        if candidate in normalized_finding or normalized_finding in candidate:
            return True
        if SequenceMatcher(None, candidate, normalized_finding).ratio() >= 0.55:
            return True
    return False


def _clauses(quote: str) -> list[str]:
    return [
        normalize(clause)
        for clause in re.split(r"[，。；;、,]", quote)
        if len(normalize(clause)) >= 4
    ]


def quote_hit(quote: str, findings: list[dict[str, Any]]) -> bool:
    normalized_quote = normalize(quote)
    if not normalized_quote:
        return False
    return any(
        _candidate_hit(candidate, findings)
        for candidate in [normalized_quote, *_clauses(quote)]
    )


def quote_full_hit(quote: str, findings: list[dict[str, Any]]) -> bool:
    clauses = _clauses(quote)
    if not clauses:
        return quote_hit(quote, findings)
    return all(_candidate_hit(clause, findings) for clause in clauses)


def category_hit(category: str, findings: list[dict[str, Any]]) -> bool:
    for finding in findings:
        rule_category = finding["category"]
        if category == rule_category or category in rule_category or rule_category in category:
            return True
    return False


def modality(suggestion: str) -> str:
    has_audio = "音频" in suggestion or "字幕" in suggestion or "口播" in suggestion
    has_visual = "画面" in suggestion or "视频画面" in suggestion
    if has_audio and has_visual:
        return "mixed"
    if has_audio:
        return "text"
    if has_visual:
        return "visual"
    return "unknown"


def select_videos(video_ids: list[str], excel: dict[str, list[ExcelRow]], limit: int) -> list[str]:
    candidates = [video_id for video_id in video_ids if video_id in excel]
    candidates.sort(
        key=lambda video_id: (
            -sum(bool(QUOTE_RE.search(row.suggestion)) for row in excel[video_id]),
            VIDEO_DIR.joinpath(f"{video_id}.mp4").stat().st_size,
        )
    )
    selected: list[str] = []
    seen_categories: set[str] = set()
    for video_id in candidates:
        categories = {row.category for row in excel[video_id]}
        if categories - seen_categories:
            selected.append(video_id)
            seen_categories.update(categories)
        if len(selected) >= limit:
            return selected
    for video_id in candidates:
        if video_id not in selected:
            selected.append(video_id)
        if len(selected) >= limit:
            break
    return selected


def load_or_transcribe(video_id: str, refresh: bool) -> dict[str, Any]:
    cache_path = ASR_DIR / f"{video_id}.json"
    if cache_path.is_file() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    video_path = VIDEO_DIR / f"{video_id}.mp4"
    try:
        result = transcribe_video(video_path, ASR_DIR / video_id)
        payload = {
            "id": video_id,
            "text": result.text,
            "duration_ms": result.duration_ms,
            "request_id": result.request_id,
            "model": result.model,
            "segments": [
                {
                    "index": segment.index,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "text": segment.text,
                }
                for segment in result.segments
            ],
        }
    except AsrError as exc:
        payload = {"id": video_id, "error": str(exc), "code": exc.code}
    ASR_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def evaluate_video(video_id: str, excel: dict[str, list[ExcelRow]], asr: dict[str, Any]) -> dict[str, Any]:
    if asr.get("error"):
        return {"id": video_id, "error": asr["error"], "rows": [], "findings": []}

    segments = [Segment(**item) for item in asr.get("segments", [])]
    transcript = text_from_segments(segments)
    review = review_transcript(
        {"has_face": None, "category": "shapewear"},
        transcript,
        segments,
    )
    findings = [
        {
            "id": finding["id"],
            "rule_id": finding["rule_id"],
            "severity": finding["severity"],
            "category": finding["category"],
            "quote": finding["quote"],
            "start_ms": finding["start_ms"],
            "end_ms": finding["end_ms"],
        }
        for finding in review.findings
    ]

    row_results: list[dict[str, Any]] = []
    for row in excel[video_id]:
        quotes = QUOTE_RE.findall(row.suggestion)
        matched_quotes = [quote for quote in quotes if quote_hit(quote, findings)]
        full_matched_quotes = [quote for quote in quotes if quote_full_hit(quote, findings)]
        hit = bool(matched_quotes) if quotes else category_hit(row.category, findings)
        full_hit = bool(full_matched_quotes) if quotes else hit
        row_results.append(
            {
                "category": row.category,
                "suggestion": row.suggestion,
                "quotes": quotes,
                "matched_quotes": matched_quotes,
                "full_matched_quotes": full_matched_quotes,
                "hit": hit,
                "full_hit": full_hit,
            }
        )
    return {
        "id": video_id,
        "duration_ms": asr.get("duration_ms"),
        "text": transcript,
        "findings": findings,
        "rows": row_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--ids", default="")
    parser.add_argument("--refresh-asr", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()

    excel = load_excel()
    video_ids = sorted(path.stem for path in VIDEO_DIR.glob("*.mp4"))
    mapping = {
        "video_files": len(video_ids),
        "text_video_candidates": sum(
            1
            for video_id in video_ids
            if any(
                modality(row.suggestion) in {"text", "mixed"}
                for row in excel.get(video_id, [])
            )
        ),
        "visual_only_video_candidates": sum(
            1
            for video_id in video_ids
            if excel.get(video_id)
            and all(modality(row.suggestion) == "visual" for row in excel[video_id])
        ),
        "excel_rows": sum(len(rows) for rows in excel.values()),
        "excel_unique_ids": len(excel),
        "matched_ids": len(set(video_ids) & set(excel)),
        "videos_without_excel": len(set(video_ids) - set(excel)),
        "excel_ids_without_video": len(set(excel) - set(video_ids)),
        "modalities": dict(
            Counter(
                modality(row.suggestion)
                for rows in excel.values()
                for row in rows
            ).most_common()
        ),
        "categories": dict(
            Counter(row.category for rows in excel.values() for row in rows).most_common()
        ),
    }
    print(json.dumps(mapping, ensure_ascii=True, indent=2))
    if args.scan_only:
        return 0

    candidate_ids = video_ids
    if args.text_only:
        candidate_ids = [
            video_id
            for video_id in video_ids
            if any(modality(row.suggestion) in {"text", "mixed"} for row in excel.get(video_id, []))
        ]
    selected = (
        [item.strip() for item in args.ids.split(",") if item.strip()]
        if args.ids
        else select_videos(candidate_ids, excel, args.limit)
    )
    selected = [video_id for video_id in selected if video_id in excel]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for video_id in selected:
        print(f"processing {video_id} ...", flush=True)
        asr = load_or_transcribe(video_id, args.refresh_asr)
        result = evaluate_video(video_id, excel, asr)
        (RESULT_DIR / f"{video_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        rows = result["rows"]
        quotes_total = sum(len(row["quotes"]) for row in rows)
        quotes_hit = sum(len(row["matched_quotes"]) for row in rows)
        quotes_full_hit = sum(len(row["full_matched_quotes"]) for row in rows)
        summary_rows.append(
            {
                "id": video_id,
                "error": result.get("error"),
                "duration_ms": result.get("duration_ms"),
                "findings": len(result.get("findings", [])),
                "rows": len(rows),
                "rows_hit": sum(1 for row in rows if row["hit"]),
                "rows_full_hit": sum(1 for row in rows if row["full_hit"]),
                "quotes_total": quotes_total,
                "quotes_hit": quotes_hit,
                "quotes_full_hit": quotes_full_hit,
            }
        )
        print(json.dumps(summary_rows[-1], ensure_ascii=True), flush=True)

    summary = {
        "mapping": mapping,
        "selected": selected,
        "videos": summary_rows,
        "totals": {
            "videos": len(summary_rows),
            "errors": sum(1 for row in summary_rows if row["error"]),
            "findings": sum(row["findings"] for row in summary_rows),
            "rows": sum(row["rows"] for row in summary_rows),
            "rows_hit": sum(row["rows_hit"] for row in summary_rows),
            "rows_full_hit": sum(row["rows_full_hit"] for row in summary_rows),
            "quotes_total": sum(row["quotes_total"] for row in summary_rows),
            "quotes_hit": sum(row["quotes_hit"] for row in summary_rows),
            "quotes_full_hit": sum(row["quotes_full_hit"] for row in summary_rows),
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["totals"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
