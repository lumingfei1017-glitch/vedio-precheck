from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.review import review_transcript  # noqa: E402
from app.semantic import SemanticError, review_semantics  # noqa: E402
from app.srt import Segment, text_from_segments  # noqa: E402
from ref_eval import load_excel, load_or_transcribe, modality, quote_full_hit, quote_hit  # noqa: E402

EVAL_DIR = ROOT / "data" / "ref-eval"
SEMANTIC_DIR = EVAL_DIR / "semantic"
RESULT_DIR = EVAL_DIR / "results"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    excel = load_excel()
    if args.ids:
        selected = [item.strip() for item in args.ids.split(",") if item.strip()]
    else:
        summary_path = EVAL_DIR / "summary.json"
        selected = json.loads(summary_path.read_text(encoding="utf-8")).get("selected", [])
    if args.limit:
        selected = selected[: args.limit]

    stats: Counter[str] = Counter()
    misses: Counter[tuple[str, str]] = Counter()
    full_misses: Counter[tuple[str, str]] = Counter()
    video_rows: list[dict[str, Any]] = []
    SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)

    for video_id in selected:
        print(f"semantic {video_id} ...", flush=True)
        asr = load_or_transcribe(video_id, refresh=False)
        if asr.get("error"):
            video_rows.append({"id": video_id, "error": asr["error"]})
            continue
        segments = [Segment(**item) for item in asr.get("segments", [])]
        transcript = text_from_segments(segments)
        cache_path = SEMANTIC_DIR / f"{video_id}.json"

        if cache_path.is_file() and not args.refresh:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            events = cached.get("events", [])
            error = cached.get("error")
        else:
            started = time.time()
            rule_result = review_transcript(
                {"has_face": None, "category": "shapewear"},
                transcript,
                segments,
            )
            try:
                semantic_events = review_semantics(
                    transcript,
                    segments,
                    rule_result.findings,
                    has_face=None,
                )
                events = [
                    {
                        "rule_id": event.rule_id,
                        "title": event.title,
                        "severity": event.severity,
                        "evidence": [
                            {
                                "quote": evidence.quote,
                                "start_ms": evidence.start_ms,
                                "end_ms": evidence.end_ms,
                            }
                            for evidence in event.evidence
                        ],
                        "reason": event.reason,
                        "suggestion": event.suggestion,
                        "method": event.method,
                        "abc_class": event.abc_class,
                        "confidence": event.confidence,
                    }
                    for event in semantic_events
                ]
                error = None
            except SemanticError as exc:
                events = []
                error = str(exc)
            cache_path.write_text(
                json.dumps({"id": video_id, "events": events, "error": error}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  {round(time.time() - started, 1)}s events={len(events)} error={error}", flush=True)

        evidence = [
            evidence
            for event in events
            for evidence in event.get("evidence", [])
        ]
        rule_result_path = RESULT_DIR / f"{video_id}.json"
        rule_rows = []
        rule_findings = []
        if rule_result_path.is_file():
            rule_payload = json.loads(rule_result_path.read_text(encoding="utf-8"))
            rule_rows = rule_payload.get("rows", [])
            rule_findings = rule_payload.get("findings", [])

        video_stats: Counter[str] = Counter()
        for row in excel.get(video_id, []):
            quotes = re.findall(r"“([^”]+)”", row.suggestion)
            if not quotes:
                continue
            kind = modality(row.suggestion)
            matched = [quote for quote in quotes if quote_hit(quote, evidence)]
            rule_matched = [quote for quote in quotes if quote_hit(quote, rule_findings)]
            combined_findings = [*rule_findings, *evidence]
            combined_matched = [
                quote for quote in quotes if quote_hit(quote, combined_findings)
            ]
            semantic_full = [quote for quote in quotes if quote_full_hit(quote, evidence)]
            rule_full = [quote for quote in quotes if quote_full_hit(quote, rule_findings)]
            combined_full = [
                quote
                for quote in quotes
                if quote_full_hit(quote, combined_findings)
            ]
            video_stats[f"{kind}_quotes"] += len(quotes)
            video_stats[f"{kind}_semantic_hit"] += len(matched)
            video_stats[f"{kind}_rule_hit"] += len(rule_matched)
            video_stats[f"{kind}_combined_hit"] += len(combined_matched)
            video_stats[f"{kind}_semantic_full"] += len(semantic_full)
            video_stats[f"{kind}_rule_full"] += len(rule_full)
            video_stats[f"{kind}_combined_full"] += len(combined_full)
            video_stats[f"{kind}_rows"] += 1
            video_stats[f"{kind}_semantic_any"] += int(bool(matched))
            video_stats[f"{kind}_semantic_majority"] += int(len(matched) * 2 >= len(quotes))
            video_stats[f"{kind}_semantic_all"] += int(len(matched) == len(quotes))
            video_stats[f"{kind}_combined_any"] += int(bool(combined_matched))
            video_stats[f"{kind}_combined_majority"] += int(len(combined_matched) * 2 >= len(quotes))
            video_stats[f"{kind}_combined_all"] += int(len(combined_matched) == len(quotes))
            for quote in quotes:
                if quote not in combined_matched:
                    misses[(kind, quote)] += 1
                if quote not in combined_full:
                    full_misses[(kind, quote)] += 1

        for key, value in video_stats.items():
            stats[key] += value
        video_rows.append(
            {
                "id": video_id,
                "events": len(events),
                "evidence": len(evidence),
                "error": error,
                **dict(video_stats),
            }
        )
        print(json.dumps(video_rows[-1], ensure_ascii=True), flush=True)

    summary = {
        "selected": selected,
        "videos": video_rows,
        "totals": dict(stats),
        "top_misses": [
            {"modality": kind, "count": count, "quote": quote}
            for (kind, quote), count in misses.most_common(30)
        ],
        "top_full_misses": [
            {"modality": kind, "count": count, "quote": quote}
            for (kind, quote), count in full_misses.most_common(30)
        ],
    }
    (EVAL_DIR / "semantic-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["totals"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
