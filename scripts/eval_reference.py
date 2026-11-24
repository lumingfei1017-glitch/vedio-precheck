from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rules import scan  # noqa: E402

REF = ROOT.parent / "参考" / "审核建议_HY-北极绒官方旗舰店_红包-乘方04-21 18.25_20261119.xlsx"
QUOTE_RE = re.compile(r"“([^”]+)”")


def main() -> int:
    workbook = openpyxl.load_workbook(REF, read_only=True, data_only=True)
    sheet = workbook["审核建议明细"]
    total = 0
    hit = 0
    misses: Counter[str] = Counter()
    hit_rules: Counter[str] = Counter()

    for row in sheet.iter_rows(min_row=2, values_only=True):
        suggestion = str(row[9] or "")
        for quote in QUOTE_RE.findall(suggestion):
            quote = quote.strip()
            if len(quote) < 3:
                continue
            total += 1
            matches = scan(quote)
            if matches:
                hit += 1
                for match in matches:
                    hit_rules[match.rule.id] += 1
            else:
                misses[quote] += 1

    print(
        json.dumps(
            {
                "quoted_phrases": total,
                "hit": hit,
                "hit_rate": round(hit / total, 3) if total else 0,
                "hit_rules": hit_rules,
                "top_misses": misses.most_common(40),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
