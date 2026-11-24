"""手动同步平台审核意见。

用法：
  .venv/Scripts/python.exe scripts/sync_audits.py
  .venv/Scripts/python.exe scripts/sync_audits.py --days 30 --kind reject
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audit_feed import AuditFeedError, recent_patterns, stats, sync_audits
from app.db import init_db


def main() -> int:
    parser = argparse.ArgumentParser(description="同步千川审核意见")
    parser.add_argument("--kind", default="reject", choices=["reject", "optimize", "all"])
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    init_db()
    kinds = ["reject", "optimize"] if args.kind == "all" else [args.kind]
    try:
        for kind in kinds:
            result = sync_audits(kind=kind, days=args.days)
            print(json.dumps(result, ensure_ascii=False, indent=1))
    except AuditFeedError as exc:
        print(f"同步失败：{exc}", file=sys.stderr)
        return 1

    print(json.dumps(stats(), ensure_ascii=False, indent=1))
    for item in recent_patterns(5):
        print(f"- [{item['count']}] {item['template'][:70]}")
        for quote in item["quotes"]:
            print(f"    例：{quote[:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
