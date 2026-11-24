from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "data" / "ref-eval" / "results"
OUT_PATH = ROOT / "docs" / "ref-eval-pilot.md"


def main() -> int:
    summary_path = ROOT / "data" / "ref-eval" / "summary.json"
    selected = set()
    if summary_path.is_file():
        selected = set(json.loads(summary_path.read_text(encoding="utf-8")).get("selected", []))
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULT_DIR.glob("*.json"))
        if not selected or path.stem in selected
    ]
    lines = ["# 参考视频评测（试点）", ""]
    for result in results:
        lines.append(f"## {result['id']}")
        lines.append(f"- 时长：{round((result.get('duration_ms') or 0) / 1000, 1)} 秒")
        for row in result.get("rows", []):
            lines.append(f"- 平台类别：{row['category']}")
            for quote in row.get("quotes", []):
                mark = "命中" if quote in row.get("matched_quotes", []) else "未命中"
                lines.append(f"  - 引文（{mark}）：{quote}")
            lines.append(f"  - 平台建议：{row['suggestion']}")
        lines.append("- 本地 findings：")
        for finding in result.get("findings", []):
            lines.append(
                f"  - {finding['rule_id']} | {finding['quote']} | "
                f"{(finding.get('start_ms') or 0) / 1000:.1f}s"
            )
        lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(results)} videos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
