"""从审核判例挖候选关键词规则。

用法：
  .venv/Scripts/python.exe scripts/mine_rules.py --limit 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import init_db
from app.rule_miner import write_mining_files


def main() -> int:
    parser = argparse.ArgumentParser(description="挖掘候选审核规则")
    parser.add_argument("--min-materials", type=int, default=3)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    init_db()
    summary = write_mining_files(
        Path(args.out) if args.out else None,
        min_materials=args.min_materials,
        limit=args.limit,
    )
    print(
        f"候选 {summary['candidates']} 条 → 阻断规则 {summary['blocking']} 条，"
        f"提示词 {summary['hints']} 条"
    )
    print(f"规则文件：{summary['rules_path']}")
    print(f"提示文件：{summary['hints_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
