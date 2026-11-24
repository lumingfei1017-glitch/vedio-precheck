from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.visual import VisualError, review_visual  # noqa: E402
from ref_eval import VIDEO_DIR, load_excel, modality, quote_full_hit, quote_hit  # noqa: E402

EVAL_DIR = ROOT / "data" / "ref-eval"
VISUAL_DIR = EVAL_DIR / "visual"
VISUAL_TERMS = [
    "低俗", "擦边", "敏感", "特写", "臀部", "胸部", "腰腹", "腹部", "透视",
    "蕾丝", "内裤", "曲线", "暴露", "紧身", "品牌", "logo", "商标", "包装",
    "混淆", "拉伸", "承重", "变形", "穿前", "穿后", "对比", "赘肉", "身材",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    excel = load_excel()
    video_ids = sorted(path.stem for path in VIDEO_DIR.glob("*.mp4"))
    candidates = [
        video_id
        for video_id in video_ids
        if excel.get(video_id)
        and all(modality(row.suggestion) == "visual" for row in excel[video_id])
    ]
    if args.ids:
        selected = [item.strip() for item in args.ids.split(",") if item.strip()]
    else:
        selected = candidates
    if args.limit:
        selected = selected[: args.limit]

    print(
        json.dumps(
            {"video_files": len(video_ids), "visual_only_candidates": len(candidates)},
            ensure_ascii=True,
        )
    )
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    misses: Counter[tuple[str, str]] = Counter()
    full_misses: Counter[tuple[str, str]] = Counter()
    video_rows = []

    for video_id in selected:
        print(f"visual {video_id} ...", flush=True)
        cache_path = VISUAL_DIR / f"{video_id}.json"
        if cache_path.is_file() and not args.refresh:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            events = cached.get("events", [])
            error = cached.get("error")
        else:
            started = time.time()
            try:
                visual_events = review_visual(
                    VIDEO_DIR / f"{video_id}.mp4",
                    work_dir=VISUAL_DIR / video_id,
                )
                events = [
                    {
                        "rule_id": event.rule_id,
                        "title": event.title,
                        "category": event.category,
                        "severity": event.severity,
                        "evidence": [
                            {
                                "quote": item.description,
                                "start_ms": item.start_ms,
                                "end_ms": item.end_ms,
                                "frame_ms": item.frame_ms,
                                "frame_index": item.frame_index,
                            }
                            for item in event.evidence
                        ],
                        "reason": event.reason,
                        "suggestion": event.suggestion,
                        "confidence": event.confidence,
                    }
                    for event in visual_events
                ]
                error = None
            except VisualError as exc:
                events = []
                error = str(exc)
            cache_path.write_text(
                json.dumps({"id": video_id, "events": events, "error": error}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"  {round(time.time() - started, 1)}s events={len(events)} error={error}",
                flush=True,
            )

        evidence = [
            item
            for event in events
            for item in event.get("evidence", [])
        ]
        descriptions = [str(item.get("quote") or "") for item in evidence]
        event_categories = [str(event.get("category") or "") for event in events]
        video_stats: Counter[str] = Counter()
        for row in excel.get(video_id, []):
            quotes = re.findall(r"“([^”]+)”", row.suggestion)
            if not quotes:
                continue
            matched = [quote for quote in quotes if quote_hit(quote, evidence)]
            full_matched = [quote for quote in quotes if quote_full_hit(quote, evidence)]
            category_match = any(
                row.category in category or category in row.category
                for category in event_categories
            )
            keyword_matches = []
            for quote in quotes:
                quote_terms = [term for term in VISUAL_TERMS if term in quote]
                matched_terms = [
                    term
                    for term in quote_terms
                    if any(term in description for description in descriptions)
                ]
                if category_match and (not quote_terms or matched_terms):
                    keyword_matches.append(quote)
            video_stats["quotes"] += len(quotes)
            video_stats["hit"] += len(matched)
            video_stats["full_hit"] += len(full_matched)
            video_stats["strict_hit"] += len(matched)
            video_stats["category_hit"] += int(category_match) * len(quotes)
            video_stats["keyword_hit"] += len(keyword_matches)
            video_stats["rows"] += 1
            video_stats["any"] += int(bool(matched))
            video_stats["majority"] += int(len(matched) * 2 >= len(quotes))
            video_stats["all"] += int(len(matched) == len(quotes))
            video_stats["category_rows"] += int(category_match)
            video_stats["keyword_rows"] += int(bool(keyword_matches))
            for quote in quotes:
                if quote not in matched:
                    misses[("visual", quote)] += 1
                if quote not in full_matched:
                    full_misses[("visual", quote)] += 1

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
            {"count": count, "quote": quote}
            for (_, quote), count in misses.most_common(30)
        ],
        "top_full_misses": [
            {"count": count, "quote": quote}
            for (_, quote), count in full_misses.most_common(30)
        ],
    }
    (EVAL_DIR / "visual-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["totals"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
