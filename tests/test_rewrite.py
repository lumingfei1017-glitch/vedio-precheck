from __future__ import annotations

import json

import app.review as review_module
import app.rewrite as rewrite_module
from app.rewrite import RewriteResult
from app.srt import Segment


def test_rewrite_script_and_cut_actions(monkeypatch):
    monkeypatch.setattr(rewrite_module, "REWRITE_ENABLED", True)
    findings = [
        {"id": "f1", "rule_id": "AP-01", "title": "量化形体改变", "severity": "blocking", "quote": "上身显瘦十几斤", "start_ms": 0, "end_ms": 3000},
        {"id": "f2", "rule_id": "AP-03", "title": "穿前穿后对比", "severity": "blocking", "quote": "穿前穿后对比", "start_ms": 3000, "end_ms": 6000},
    ]
    suggestions = [
        {"id": "s1", "finding_id": "f1", "before_text": "上身显瘦十几斤", "after_text": "旧建议", "method": "TTS_SUBTITLE", "abc_class": "A", "original_duration_ms": 3000},
        {"id": "s2", "finding_id": "f2", "before_text": "穿前穿后对比", "after_text": "旧建议", "method": "CUT_BROLL", "abc_class": "B", "original_duration_ms": 3000},
    ]
    response = {
        "rewrites": [
            {
                "target_index": 0,
                "action": "replace_script",
                "after_text": "姐妹们，这件上身包裹感挺明显，腰腹这块看着更利落。",
                "edit_instruction": "替换原口播",
                "kept_selling_point": "腰腹包裹感",
                "estimated_duration_ms": 2800,
                "alternatives": ["这件版型挺利落。"],
                "confidence": 0.9,
            },
            {
                "target_index": 1,
                "action": "replace_script",
                "after_text": "这里会被强制改成剪辑指令",
                "edit_instruction": "删除穿前穿后对比镜头",
                "kept_selling_point": "版型",
                "estimated_duration_ms": None,
                "alternatives": [],
                "confidence": 0.8,
            },
        ]
    }
    monkeypatch.setattr(
        rewrite_module,
        "_call_rewrite",
        lambda api_key, payload: json.dumps(response, ensure_ascii=False),
    )
    segments = [
        Segment(1, 0, 3000, "上身显瘦十几斤"),
        Segment(2, 3000, 6000, "穿前穿后对比"),
    ]
    results = rewrite_module.rewrite_suggestions(
        "上身显瘦十几斤\n穿前穿后对比",
        segments,
        findings,
        suggestions,
    )
    assert results["s1"].action == "replace_script"
    assert "包裹感" in (results["s1"].after_text or "")
    assert results["s2"].action == "cut_segment"
    assert results["s2"].after_text is None


def test_unsafe_rewrite_is_rejected(monkeypatch):
    monkeypatch.setattr(rewrite_module, "REWRITE_ENABLED", True)
    findings = [
        {"id": "f1", "rule_id": "AP-01", "title": "量化形体改变", "severity": "blocking", "quote": "上身显瘦十几斤", "start_ms": 0, "end_ms": 3000}
    ]
    suggestions = [
        {"id": "s1", "finding_id": "f1", "before_text": "上身显瘦十几斤", "after_text": "旧建议", "method": "TTS_SUBTITLE", "abc_class": "A", "original_duration_ms": 3000}
    ]
    response = {
        "rewrites": [
            {
                "target_index": 0,
                "action": "replace_script",
                "after_text": "这件上身显瘦十几斤。",
                "edit_instruction": "",
                "kept_selling_point": "",
                "estimated_duration_ms": 2000,
                "alternatives": [],
                "confidence": 0.9,
            }
        ]
    }
    monkeypatch.setattr(
        rewrite_module,
        "_call_rewrite",
        lambda api_key, payload: json.dumps(response, ensure_ascii=False),
    )
    results = rewrite_module.rewrite_suggestions(
        "上身显瘦十几斤",
        [Segment(1, 0, 3000, "上身显瘦十几斤")],
        findings,
        suggestions,
    )
    assert results == {}


def test_rewrite_is_applied_to_review(monkeypatch):
    monkeypatch.setattr(review_module, "SEMANTIC_ENABLED", False)
    monkeypatch.setattr(review_module, "VISUAL_ENABLED", False)
    monkeypatch.setattr(review_module, "REWRITE_ENABLED", True)

    def fake_rewrite(transcript, segments, findings, suggestions, **kwargs):
        return {
            suggestions[0]["id"]: RewriteResult(
                action="replace_script",
                after_text="姐妹们，这件上身包裹感挺明显。",
                edit_instruction="替换原口播",
                kept_selling_point="包裹感",
                estimated_duration_ms=2200,
                alternatives=("这件版型挺利落。",),
                confidence=0.9,
            )
        }

    monkeypatch.setattr(review_module, "rewrite_suggestions", fake_rewrite)
    result = review_module.review_material_content(
        {"has_face": 0, "category": "shapewear"},
        "姐妹们，上身显瘦十几斤。",
        [Segment(1, 0, 3000, "姐妹们，上身显瘦十几斤。")],
    )
    assert result.suggestions[0]["after_text"] == "姐妹们，这件上身包裹感挺明显。"
    assert result.suggestions[0]["action"] == "replace_script"
    assert result.summary["rewrite_count"] == 1
from app.rewrite import _parse_rewrite


def test_parse_rewrite_swaps_in_longest_fitting_alternative():
    item = {
        "action": "replace_script",
        "after_text": "主" * 30,
        "alternatives": ["短" * 12, "中" * 20],
        "estimated_duration_ms": 9000,
    }
    suggestion = {
        "id": "s1",
        "method": "WAN_LIPSYNC",
        "before_text": "原句",
        "original_duration_ms": 5520,
    }

    result = _parse_rewrite(item, suggestion)

    assert result is not None
    assert result.after_text == "中" * 20
    assert result.estimated_duration_ms == 4000


def test_parse_rewrite_keeps_main_text_when_it_fits():
    item = {
        "action": "replace_script",
        "after_text": "主" * 20,
        "alternatives": ["短" * 12],
        "estimated_duration_ms": 4000,
    }
    suggestion = {
        "id": "s1",
        "method": "WAN_LIPSYNC",
        "before_text": "原句",
        "original_duration_ms": 5520,
    }

    result = _parse_rewrite(item, suggestion)

    assert result is not None
    assert result.after_text == "主" * 20
