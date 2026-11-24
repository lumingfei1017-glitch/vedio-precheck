from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.review import review_transcript  # noqa: E402
from app.semantic import review_semantics  # noqa: E402
from app.srt import Segment, parse_srt, read_text, text_from_segments  # noqa: E402


def main() -> int:
    material_id = sys.argv[1] if len(sys.argv) > 1 else "5a4d771391a5"
    if material_id.startswith("ref:"):
        ref_id = material_id[4:]
        cache_path = ROOT / "data" / "ref-eval" / "asr" / f"{ref_id}.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("error"):
            print(json.dumps(cached, ensure_ascii=False))
            return 1
        segments = [Segment(**item) for item in cached["segments"]]
        transcript = text_from_segments(segments)
        material = {"has_face": None, "category": "shapewear"}
    else:
        conn = sqlite3.connect(ROOT / "data" / "app.db")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM material WHERE id = ?", (material_id,)).fetchone()
        if row is None:
            print(f"material not found: {material_id}")
            return 1
        material = dict(row)
        segments = []
        if material.get("srt_path") and Path(material["srt_path"]).is_file():
            segments = parse_srt(read_text(Path(material["srt_path"])))
        transcript = material.get("transcript") or ""
    rule_result = review_transcript(material, transcript, segments)
    events = review_semantics(
        transcript,
        segments,
        rule_result.findings,
        has_face=bool(material["has_face"]) if material["has_face"] is not None else None,
    )
    output = [
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
                for event in events
            ]
    (ROOT / "data" / "semantic-smoke.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
