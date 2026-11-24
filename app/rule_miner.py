"""从平台拒审判例里挖候选关键词规则。

思路：意见模板太粗（878 条都是"夸大功效"），但括号里的违规原句很具体。
对原句做 n-gram 统计，按"覆盖多少个不同素材/账户"排序，再用功效类词素过滤，
得到一批有证据支撑的候选词，交给人工一键采纳或放进影子模式观察。

注意：这里只产出候选，不自动写入 rules/qianchuan_apparel.json。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import GENERATED_RULES_PATH, PROMPT_HINTS_PATH
from .db import get_db

_TZ = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "rules" / "candidates.json"

# 塑身衣品类的功效词素：候选词必须至少命中一个，过滤掉品牌名和中性名词。
CLAIM_HINTS = (
    "收", "紧", "瘦", "提", "塑", "压", "平", "细", "翘",
    "减", "燃", "脂", "赘肉", "曲线", "管理", "锁", "藏", "兜",
)

# 中性词/产品名：单独出现不构成违规，不能做硬规则。
NEUTRAL_GRAMS = {
    "赘肉", "脂肪", "赘肉脂肪", "身材", "曲线", "塑身", "塑形",
    "塑身衣", "塑身裤", "收腹裤", "提臀裤", "收腹", "提臀",
}
PRODUCT_WORDS = ("收腹裤", "塑身衣", "塑身裤", "提臀裤", "身材管理器")
# 以虚词开头/结尾的 n-gram 基本都是切碎的片段。
EDGE_STOP = set("的了是又都就把给我你他她它这那个一不没很更最还")

PUNCT = re.compile(r"[\s，。！？；：、,.!?;:：（）()\"“”「」]")


def _clean(text: str) -> str:
    return PUNCT.sub("", text or "")


def _load_quotes(kind: str = "reject") -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT api_key, adv, material_id, template, quotes_json, last_seen
            FROM audit_case
            WHERE kind = ? AND quotes_json IS NOT NULL AND quotes_json <> '[]'
            """,
            (kind,),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            quotes = json.loads(row["quotes_json"])
        except (TypeError, ValueError):
            continue
        for quote in quotes:
            text = _clean(str(quote))
            if 2 <= len(text) <= 60:
                items.append(
                    {
                        "text": text,
                        "api_key": row["api_key"],
                        "adv": row["adv"],
                        "material_id": row["material_id"],
                        "template": row["template"],
                        "last_seen": row["last_seen"],
                    }
                )
    return items


def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度，用于判断两个候选是否只是同一句的不同切片。"""
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _ngrams(text: str, low: int = 2, high: int = 8) -> Iterable[str]:
    seen: set[str] = set()
    for size in range(low, high + 1):
        for start in range(0, len(text) - size + 1):
            gram = text[start : start + size]
            if gram not in seen:
                seen.add(gram)
                yield gram


def mine_candidates(
    *,
    min_materials: int = 3,
    limit: int = 40,
    low: int = 2,
    high: int = 8,
) -> list[dict[str, Any]]:
    quotes = _load_quotes("reject")
    # 已通过但待优化的样本当负样本，用来估计误报。
    passed = _load_quotes("optimize")
    passed_support: dict[str, set[str]] = defaultdict(set)
    for item in passed:
        for gram in _ngrams(item["text"], low, high):
            passed_support[gram].add(item["material_id"] or item["api_key"])

    support: dict[str, set[str]] = defaultdict(set)
    advs: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[str]] = defaultdict(list)
    last_seen: dict[str, str] = {}

    for item in quotes:
        for gram in _ngrams(item["text"], low, high):
            support[gram].add(item["material_id"] or item["api_key"])
            if item["adv"]:
                advs[gram].add(item["adv"])
            if item["text"] not in examples[gram] and len(examples[gram]) < 3:
                examples[gram].append(item["text"])
            stamp = item["last_seen"] or ""
            if stamp > last_seen.get(gram, ""):
                last_seen[gram] = stamp

    ranked = sorted(
        (
            (gram, len(keys))
            for gram, keys in support.items()
            if len(keys) >= min_materials
        ),
        key=lambda pair: (-pair[1], -len(pair[0])),
    )

    kept: list[str] = []
    for gram, _count in ranked:
        if any(gram in existing for existing in kept):
            continue
        if not any(hint in gram for hint in CLAIM_HINTS):
            continue
        if gram in NEUTRAL_GRAMS:
            continue
        if any(gram in neutral for neutral in NEUTRAL_GRAMS):
            continue
        if gram[0] in EDGE_STOP or gram[-1] in EDGE_STOP:
            continue
        # 以数量词/方位词结尾的多半是被截断的片段，如"让你轻松显瘦十几"。
        if gram[-1] in "十百千万几来到上下里中":
            continue
        if gram.startswith(("下来", "起来", "出来", "进去", "上去", "下来", "一下")):
            continue
        if any(word in gram for word in PRODUCT_WORDS):
            continue
        if any(gram in word for word in PRODUCT_WORDS):
            continue
        kept.append(gram)
        if len(kept) >= limit * 2:
            break

    # 后处理：去掉被更长候选包含、且支持度接近的碎片（如"显瘦十" vs "显瘦十几斤"），
    # 以及同一句切出来、高度重叠的片段（保留先出现的那条）。
    final: list[str] = []
    for gram in kept:
        redundant = any(
            gram != other
            and len(support[other]) >= 0.6 * len(support[gram])
            and (
                gram in other
                or (
                    len(other) >= len(gram)
                    and _lcs_len(gram, other) >= max(4, int(0.6 * len(gram)))
                )
            )
            for other in final
        )
        if not redundant:
            final.append(gram)
        if len(final) >= limit:
            break
    kept = final

    now = datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S")
    candidates: list[dict[str, Any]] = []
    for gram in kept:
        reject_n = len(support[gram])
        passed_n = len(passed_support.get(gram, ()))
        total = reject_n + passed_n
        precision = reject_n / total if total else 0.0
        candidates.append(
            {
                "pattern": gram,
                "type": "keyword",
                "materials": reject_n,
                "accounts": len(advs[gram]),
                "passed_materials": passed_n,
                "precision_estimate": round(precision, 3),
                "examples": examples[gram],
                "last_seen": last_seen.get(gram),
                "generated_at": now,
                "status": "candidate",
            }
        )
    candidates.sort(key=lambda item: (-item["precision_estimate"], -item["materials"]))
    return candidates


def write_candidates(path: Path | None = None, **kwargs: Any) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_OUTPUT
    candidates = mine_candidates(**kwargs)
    payload = {
        "generated_at": datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(candidates),
        "candidates": candidates,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


QUANT_MARKERS = ("斤", "厘米", "cm", "圈", "十几", "几十", "超模", "蜜桃", "蚂蚁", "s型")


def classify_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """分两类：可直接做阻断词的，和只能进提示词的。

    阻断要求：精度估计 ≥0.9、覆盖 ≥5 个素材，且要么带量化标记（斤/厘米/圈），
    要么是 5 字以上的长短语（歧义低）。其余一律只做提示。
    """
    compound = re.compile(r"(收腹|提臀|塑形|紧致|显瘦|收腰).*(收腹|提臀|塑形|紧致|显瘦|收腰)")
    blocking: list[dict[str, Any]] = []
    hints: list[dict[str, Any]] = []
    for item in candidates:
        pattern = str(item.get("pattern") or "")
        specific = (
            len(pattern) >= 6
            or any(marker in pattern.lower() for marker in QUANT_MARKERS)
            or bool(compound.search(pattern))
        )
        if (
            float(item.get("precision_estimate") or 0) >= 0.9
            and int(item.get("materials") or 0) >= 5
            and specific
        ):
            blocking.append(item)
        else:
            hints.append(item)
    return blocking, hints


def write_mining_files(
    path: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """挖掘 → 写阻断规则文件 + 提示词文件。"""
    payload = write_candidates(path, **kwargs)
    blocking, hints = classify_candidates(payload["candidates"])

    rules = []
    for index, item in enumerate(blocking, start=1):
        rules.append(
            {
                "id": f"MK-{index:02d}",
                "title": f"判例高频表述：{item['pattern']}",
                "category": "夸大功效/虚假塑身效果",
                "severity": "blocking",
                "action": "rewrite",
                "default_method": "WAN_LIPSYNC",
                "modality": ["audio", "subtitle"],
                "patterns": [re.escape(item["pattern"])],
                "negative_patterns": [],
                "why": (
                    f"平台拒审原句高频出现（{item['materials']} 个素材 / "
                    f"{item['accounts']} 个账户），由判例自动挖掘。"
                ),
                "rewrite": {
                    "replacements": [],
                    "fallback": "改成穿着体感描述，删除确定性的塑形效果承诺。",
                },
                "required_evidence": "",
            }
        )
    generated = {
        "version": f"mined-{datetime.now(_TZ).strftime('%Y%m%d')}",
        "source": "audit_feed",
        "rules": rules,
    }
    GENERATED_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_RULES_PATH.write_text(
        json.dumps(generated, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    PROMPT_HINTS_PATH.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "hints": [item["pattern"] for item in hints],
                "detail": hints,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return {
        "candidates": payload["count"],
        "blocking": len(blocking),
        "hints": len(hints),
        "rules_path": str(GENERATED_RULES_PATH),
        "hints_path": str(PROMPT_HINTS_PATH),
    }
