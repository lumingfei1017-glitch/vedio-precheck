from __future__ import annotations

import json

import app.review as review_module
import app.semantic as semantic_module
from app.semantic import SemanticError, SemanticEvent, SemanticEvidence
from app.srt import Segment


TRANSCRIPT = (
    "无特效无剪辑，现场给姐妹看一下这个收腹效果\n"
    "肚子一下就兜进去\n"
    "压得特别紧实\n"
    "从肩膀到大腿，整个人的身材都能大变样"
)

SEGMENTS = [
    Segment(1, 1200, 4200, "无特效无剪辑，现场给姐妹看一下这个收腹效果"),
    Segment(2, 4300, 5600, "肚子一下就兜进去"),
    Segment(3, 6100, 7200, "压得特别紧实"),
    Segment(4, 8200, 10500, "从肩膀到大腿，整个人的身材都能大变样"),
]


def test_semantic_event_is_validated_against_transcript(monkeypatch):
    monkeypatch.setattr(semantic_module, "SEMANTIC_ENABLED", True)
    response = {
        "events": [
            {
                "rule_id": "SEM-APPAREL-EFFECT",
                "title": "现场演示 + 确定性塑形承诺",
                "severity": "blocking",
                "evidence": [
                    {
                        "quote": "无特效无剪辑，现场给姐妹看一下这个收腹效果",
                        "start_ms": 1200,
                        "end_ms": 4200,
                    },
                    {"quote": "肚子一下就兜进去", "start_ms": 4300, "end_ms": 5600},
                    {"quote": "这句不在原文里", "start_ms": 0, "end_ms": 100},
                ],
                "reason": "真实性背书叠加现场演示",
                "suggestion": "删除现场演示和确定性效果承诺",
                "method": "CUT_BROLL",
                "abc_class": "B",
                "confidence": 0.95,
                "needs_human": True,
            }
        ]
    }
    monkeypatch.setattr(
        semantic_module,
        "_call_model",
        lambda api_key, payload: json.dumps(response, ensure_ascii=False),
    )
    events = semantic_module.review_semantics(TRANSCRIPT, SEGMENTS, [], has_face=False)
    assert len(events) == 1
    assert len(events[0].evidence) == 2
    assert all(item.quote in TRANSCRIPT for item in events[0].evidence)


def test_semantic_event_suppresses_covered_rule_findings(monkeypatch):
    monkeypatch.setattr(review_module, "SEMANTIC_ENABLED", True)
    event = SemanticEvent(
        rule_id="SEM-APPAREL-EFFECT",
        title="现场演示 + 确定性塑形承诺",
        severity="blocking",
        evidence=(
            SemanticEvidence("无特效无剪辑，现场给姐妹看一下这个收腹效果", 1200, 4200),
            SemanticEvidence("肚子一下就兜进去", 4300, 5600),
        ),
        reason="真实性背书叠加现场演示",
        suggestion="删除现场演示和确定性效果承诺",
        method="CUT_BROLL",
        abc_class="B",
        confidence=0.95,
        needs_human=True,
    )
    monkeypatch.setattr(review_module, "review_semantics", lambda *args, **kwargs: [event])
    material = {"has_face": 0, "category": "shapewear"}
    result = review_module.review_material_content(material, TRANSCRIPT, SEGMENTS)
    semantic_findings = [finding for finding in result.findings if finding.get("source") == "semantic"]
    assert len(semantic_findings) == 1
    assert len(semantic_findings[0]["evidence"]) == 2
    assert result.summary["semantic_count"] == 1


def test_semantic_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setattr(review_module, "SEMANTIC_ENABLED", True)

    def fail(*args, **kwargs):
        raise SemanticError("model unavailable")

    monkeypatch.setattr(review_module, "review_semantics", fail)
    material = {"has_face": 0, "category": "shapewear"}
    result = review_module.review_material_content(material, TRANSCRIPT, SEGMENTS)
    assert result.summary["semantic_status"] == "failed"
    assert result.findings


def test_parse_json_tolerates_trailing_content():
    import app.semantic as semantic_module

    data = semantic_module._parse_json('{"events": [{"title": "A"}]}\n{"events": []}')

    assert data["events"][0]["title"] == "A"


def test_parse_reason_extracts_template_and_quotes():
    from app.audit_feed import parse_reason

    text = (
        "您好，服装配饰广告内容不得夸大产品的功效，请修改，谢谢。"
        " (广告视频中存在“你看我肚子都没有勒出来肉肉了”和“压的特别紧实”，因此不通过。)"
    )

    template, detail, quotes = parse_reason(text)

    assert "服装配饰广告内容不得夸大产品的功效" in template
    assert "请修改" not in template
    assert "你看我肚子都没有勒出来肉肉了" in quotes
    assert "压的特别紧实" in quotes
    assert "因此不通过" in detail


def test_sync_audits_upserts_and_builds_prompt_block(monkeypatch):
    from app import audit_feed
    from app.db import get_db, init_db

    monkeypatch.setattr(audit_feed, "AUDIT_FEED_ENABLED", True)
    init_db()
    with get_db() as conn:
        conn.execute("DELETE FROM audit_case")
        conn.execute("DELETE FROM audit_sync")
    page = {
        "ok": True,
        "total": 1,
        "size": 200,
        "items": [
            {
                "key": "case-1",
                "kind": "reject",
                "material_id": "m1",
                "reasons": [
                    "您好，服装配饰广告内容不得夸大产品的功效，请修改，谢谢。 (视频中存在“肚子一下就兜进去”，因此不通过。)"
                ],
                "first_seen": "2026-11-24 09:00:00",
                "last_seen": "2026-11-24 09:00:00",
            }
        ],
    }
    monkeypatch.setattr(audit_feed, "fetch_page", lambda **kwargs: page)

    result = audit_feed.sync_audits()
    assert result["inserted"] >= 1

    audit_feed.sync_audits()
    again = audit_feed.stats()
    assert again["cases"] >= 1

    block = audit_feed.build_prompt_block()
    assert "肚子一下就兜进去" in block


def test_mine_candidates_from_reject_quotes():
    from app.db import get_db, init_db
    from app.rule_miner import mine_candidates

    init_db()
    with get_db() as conn:
        conn.execute("DELETE FROM audit_case")
        for index in range(3):
            conn.execute(
                """
                INSERT INTO audit_case (id, api_key, kind, material_id, quotes_json, synced_at)
                VALUES (?, ?, 'reject', ?, ?, '2026-11-24 10:00:00')
                """,
                (f"k{index}", f"k{index}", f"m{index}", '["让你轻松显瘦十几斤"]'),
            )

    candidates = mine_candidates(min_materials=2, limit=20)
    patterns = [item["pattern"] for item in candidates]

    assert any("显瘦十几斤" in pattern for pattern in patterns)
    assert all(item["materials"] >= 2 for item in candidates)
