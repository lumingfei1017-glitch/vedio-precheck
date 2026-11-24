"""平台审核意见同步。

接口只读，返回千川全账户的拒审 / 待优化意见。每条意见的括号明细里通常带
引号包裹的违规原句，这部分直接作为语义审核的判例锚点，比静态 Excel 更新。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .config import (
    AUDIT_FEED_BASE_URL,
    AUDIT_FEED_ENABLED,
    AUDIT_FEED_MAX_PAGES,
    AUDIT_FEED_PAGE_SIZE,
    AUDIT_FEED_PROMPT_CASES,
    AUDIT_FEED_TIMEOUT_SECONDS,
)
from .db import get_db

_TZ = timezone(timedelta(hours=8))


class AuditFeedError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S")


def parse_reason(text: str) -> tuple[str, str, list[str]]:
    """拆成（意见模板, 括号明细, 违规原句）。"""
    raw = (text or "").strip()
    body = re.sub(r"^您好[，,]\s*", "", raw)
    detail = ""
    match = re.search(r"[（(]([^（()）]*)[）)]\s*$", body)
    if match:
        detail = match.group(1).strip()
        body = body[: match.start()].strip()
    quotes = [
        item.strip()
        for item in re.findall(r"[“\"「]([^”\"」]{2,60})[”\"」]", detail)
        if item.strip()
    ]
    template = re.sub(r"请修改[，,]?谢谢[。.]?$", "", body).strip().rstrip("。")
    return template, detail, quotes


def fetch_page(
    *,
    kind: str = "reject",
    page: int = 1,
    size: int = AUDIT_FEED_PAGE_SIZE,
    days: int | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"kind": kind, "page": page, "size": size}
    if days:
        params["days"] = days
    http = client or httpx.Client(timeout=AUDIT_FEED_TIMEOUT_SECONDS)
    try:
        response = http.get(f"{AUDIT_FEED_BASE_URL}/api/audits", params=params)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPError as exc:
        raise AuditFeedError(f"审核意见接口请求失败：{type(exc).__name__}") from exc
    finally:
        if client is None:
            http.close()
    if not isinstance(body, dict) or not body.get("ok"):
        raise AuditFeedError("审核意见接口返回异常")
    return body


def sync_audits(
    *,
    kind: str = "reject",
    days: int | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """全量拉取并增量入库，返回同步统计。"""
    if not AUDIT_FEED_ENABLED:
        return {"enabled": False, "inserted": 0, "updated": 0, "total": 0}
    inserted = updated = total = 0
    rows: list[tuple] = []
    now = _now()
    for page in range(1, AUDIT_FEED_MAX_PAGES + 1):
        body = fetch_page(kind=kind, page=page, days=days, client=client)
        total = int(body.get("total") or 0)
        items = body.get("items") or []
        if not items:
            break
        for item in items:
            api_key = str(item.get("key") or item.get("material_id") or "")
            for index, reason in enumerate(item.get("reasons") or []):
                template, detail, quotes = parse_reason(str(reason))
                rows.append(
                    (
                        f"{api_key}:{index}",
                        api_key,
                        str(item.get("kind") or kind),
                        str(item.get("adv") or ""),
                        str(item.get("adv_name") or ""),
                        str(item.get("material_id") or ""),
                        str(item.get("video_id") or ""),
                        str(item.get("filename") or ""),
                        str(item.get("platform") or ""),
                        template,
                        detail,
                        json.dumps(quotes, ensure_ascii=False),
                        str(item.get("first_seen") or ""),
                        str(item.get("last_seen") or ""),
                        now,
                    )
                )
        if page * int(body.get("size") or AUDIT_FEED_PAGE_SIZE) >= total:
            break

    with get_db() as conn:
        for row in rows:
            exists = conn.execute("SELECT 1 FROM audit_case WHERE id = ?", (row[0],)).fetchone()
            conn.execute(
                """
                INSERT INTO audit_case (id, api_key, kind, adv, adv_name, material_id, video_id,
                    filename, platform, template, detail, quotes_json, first_seen, last_seen, synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    detail = excluded.detail,
                    quotes_json = excluded.quotes_json,
                    template = excluded.template,
                    synced_at = excluded.synced_at
                """,
                row,
            )
            if exists:
                updated += 1
            else:
                inserted += 1
        conn.execute(
            """
            INSERT INTO audit_sync (id, last_synced_at, total, inserted, updated)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_synced_at = excluded.last_synced_at,
                total = excluded.total,
                inserted = excluded.inserted,
                updated = excluded.updated
            """,
            (now, total, inserted, updated),
        )
    return {
        "enabled": True,
        "kind": kind,
        "total": total,
        "fetched": len(rows),
        "inserted": inserted,
        "updated": updated,
        "synced_at": now,
    }


def stats() -> dict[str, Any]:
    with get_db() as conn:
        case_count = conn.execute("SELECT COUNT(*) FROM audit_case").fetchone()[0]
        pattern_count = conn.execute(
            "SELECT COUNT(DISTINCT template) FROM audit_case"
        ).fetchone()[0]
        row = conn.execute("SELECT * FROM audit_sync WHERE id = 1").fetchone()
    return {
        "cases": case_count,
        "patterns": pattern_count,
        "last_synced_at": row["last_synced_at"] if row else None,
        "total": row["total"] if row else 0,
    }


def recent_patterns(limit: int = AUDIT_FEED_PROMPT_CASES) -> list[dict[str, Any]]:
    """按出现次数取最近的几类拒审口径，附带引号原句。"""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT template, COUNT(*) AS n, MAX(last_seen) AS last_seen
            FROM audit_case
            WHERE kind = 'reject' AND COALESCE(template, '') <> ''
            GROUP BY template
            ORDER BY n DESC, last_seen DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        patterns: list[dict[str, Any]] = []
        for row in rows:
            quotes = [
                item[0]
                for item in conn.execute(
                    """
                    SELECT quotes_json FROM audit_case
                    WHERE template = ? AND quotes_json IS NOT NULL AND quotes_json <> '[]'
                    ORDER BY last_seen DESC LIMIT 40
                    """,
                    (row["template"],),
                ).fetchall()
            ]
            flattened: list[str] = []
            for payload in quotes:
                try:
                    values = json.loads(payload)
                except (TypeError, ValueError):
                    continue
                for value in values:
                    text = str(value).strip()
                    if not _usable_quote(text):
                        continue
                    if text not in flattened:
                        flattened.append(text)
            patterns.append(
                {
                    "template": row["template"],
                    "count": row["n"],
                    "last_seen": row["last_seen"],
                    "quotes": flattened[:3],
                }
            )
    return patterns


def build_prompt_block(limit: int = AUDIT_FEED_PROMPT_CASES) -> str:
    """给语义审核用的最新判例摘要，控制在十行以内。"""
    try:
        patterns = recent_patterns(limit)
    except Exception:  # noqa: BLE001
        return ""
    if not patterns:
        return ""
    lines: list[str] = []
    for item in patterns:
        template = item["template"].rstrip("，,、；;：: ")[:60]
        line = f"- {template}（近期 {item['count']} 条）"
        if item["quotes"]:
            samples = "／".join(f"“{q[:26]}”" for q in item["quotes"][:2])
            line += f" 例：{samples}"
        lines.append(line)
    try:
        from .rules import prompt_hints

        hints = prompt_hints()
    except Exception:  # noqa: BLE001
        hints = []
    if hints:
        lines.append("高频违规表述（仅作提示，不单独判违规）：" + "、".join(hints[:20]))
    return "\n".join(lines)


def _usable_quote(text: str) -> bool:
    """过滤 HTML 碎片、标记残留和过短过长的噪声。"""
    if len(text) < 2 or len(text) > 80:
        return False
    if re.search(r"[<>]|img\s+src|https?://|&[a-z]+;", text, flags=re.IGNORECASE):
        return False
    return True


def needs_sync(hours: float) -> bool:
    with get_db() as conn:
        row = conn.execute("SELECT last_synced_at FROM audit_sync WHERE id = 1").fetchone()
    if not row or not row["last_synced_at"]:
        return True
    try:
        last = datetime.strptime(row["last_synced_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_TZ)
    except ValueError:
        return True
    return datetime.now(_TZ) - last > timedelta(hours=hours)
