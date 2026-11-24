from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "ref-eval"
RESULT_DIR = EVAL_DIR / "results"


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


def main() -> int:
    summary = json.loads((EVAL_DIR / "summary.json").read_text(encoding="utf-8"))
    selected = summary.get("selected", [])
    stats: Counter[str] = Counter()
    misses: Counter[tuple[str, str]] = Counter()

    for video_id in selected:
        path = RESULT_DIR / f"{video_id}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        for row in result.get("rows", []):
            kind = modality(row["suggestion"])
            for quote in row.get("quotes", []):
                stats[f"{kind}_total"] += 1
                if quote in row.get("matched_quotes", []):
                    stats[f"{kind}_hit"] += 1
                if quote in row.get("full_matched_quotes", []):
                    stats[f"{kind}_full_hit"] += 1
                else:
                    misses[(kind, quote)] += 1

    for kind in ("text", "mixed", "visual", "unknown"):
        total = stats[f"{kind}_total"]
        hit = stats[f"{kind}_hit"]
        print(
            json.dumps(
                {
                    "modality": kind,
                    "quotes": total,
                    "hit": hit,
                    "full_hit": stats[f"{kind}_full_hit"],
                    "recall": round(hit / total, 3) if total else None,
                    "full_recall": round(stats[f"{kind}_full_hit"] / total, 3) if total else None,
                },
                ensure_ascii=True,
            )
        )

    print("TOP_TEXT_MISSES")
    for (kind, quote), count in misses.most_common(30):
        if kind in {"text", "mixed"}:
            print(json.dumps({"modality": kind, "count": count, "quote": quote}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
