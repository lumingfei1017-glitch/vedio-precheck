from __future__ import annotations

import json
from pathlib import Path

import app.review as review_module
import app.visual as visual_module
from app.visual import (
    VisualEvent,
    VisualEvidence,
    _cluster_spans,
    _event_with_spans,
    _locator_text,
)


def _event(start: int, end: int, description: str = "画面出现双 G 图案") -> VisualEvent:
    return VisualEvent(
        rule_id="VIS-BRAND",
        title="品牌混淆",
        category="未授权品牌/商标混淆",
        severity="suggest",
        evidence=(
            VisualEvidence(
                description=description,
                start_ms=start,
                end_ms=end,
                frame_ms=(start + end) // 2,
                frame_index=4,
            ),
        ),
        reason="",
        suggestion="",
        confidence=0.8,
    )


def test_dense_frame_indexes_map_back_to_timeline():
    frames = [(15000 + index * 500, Path(f"r_{index:03d}.jpg")) for index in range(12)]

    spans = _cluster_spans(frames, [6, 7], 60000)

    assert spans == [(18000, 19000)]


def test_locator_returns_every_occurrence():
    frames = [(13000 + index * 1000, Path(f"r_{index:03d}.jpg")) for index in range(24)]

    spans = _cluster_spans(frames, [13, 20, 21], 60000)

    assert spans == [(26000, 27000), (33000, 35000)]


def test_locator_uses_rule_hint_not_model_description():
    event = _event(0, 1000, "黑色塑身衣内侧米色蕾丝部分")

    text = _locator_text(event)

    assert "logo" in text or "老花" in text
    assert "内侧" not in text


def test_relocalized_event_replaces_original_window():
    event = _event(22483, 28103, "米色蕾丝双 G 图案")

    moved = _event_with_spans(event, [(25300, 26500), (32750, 34000)])

    assert [item.start_ms for item in moved.evidence] == [25300, 32750]
    assert moved.evidence[1].end_ms == 34000
    assert "双 G" in moved.evidence[0].description
    assert moved.rule_id == "VIS-BRAND"


def test_visual_event_is_validated_and_timed(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(visual_module, "VISUAL_ENABLED", True)
    monkeypatch.setattr(visual_module, "probe_duration_ms", lambda _: 20000)
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"not-a-real-jpeg")
    monkeypatch.setattr(
        visual_module,
        "_extract_frames",
        lambda video_path, duration, frame_dir: [(5000, frame_path), (12000, frame_path)],
    )
    response = {
        "events": [
            {
                "rule_id": "VIS-BODY-COMPARISON",
                "title": "穿前穿后对比",
                "category": "夸大功效/虚假塑身效果",
                "severity": "blocking",
                "frame_index": 1,
                "description": "画面左侧展示腹部，右侧展示穿着后效果。",
                "reason": "形成穿前穿后身材对比。",
                "suggestion": "删除对比画面。",
                "confidence": 0.92,
            },
            {
                "rule_id": "VIS-RISK",
                "title": "无效帧",
                "category": "其他视觉风险",
                "severity": "blocking",
                "frame_index": 99,
                "description": "无效",
                "reason": "无效",
                "suggestion": "无效",
                "confidence": 0.5,
            },
        ]
    }
    monkeypatch.setattr(
        visual_module,
        "_call_visual",
        lambda api_key, payload: json.dumps(response, ensure_ascii=False),
    )
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    events = visual_module.review_visual(video_path)
    assert len(events) == 1
    assert events[0].rule_id == "VIS-BODY-COMPARISON"
    assert events[0].evidence[0].frame_ms == 12000
    assert events[0].evidence[0].start_ms == 7000
    assert events[0].evidence[0].end_ms == 17000


def test_visual_event_merges_into_review(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    event = VisualEvent(
        rule_id="VIS-LOW-SUGGESTIVE",
        title="臀部特写",
        category="不良导向/低俗擦边",
        severity="blocking",
        evidence=(
            VisualEvidence(
                description="画面是臀部局部特写。",
                start_ms=3000,
                end_ms=5000,
                frame_ms=4000,
                frame_index=2,
            ),
        ),
        reason="局部特写突出敏感部位。",
        suggestion="删除该镜头。",
        confidence=0.9,
    )
    monkeypatch.setattr(review_module, "SEMANTIC_ENABLED", False)
    monkeypatch.setattr(review_module, "VISUAL_ENABLED", True)
    monkeypatch.setattr(review_module, "review_visual", lambda path: [event])
    result = review_module.review_material_content(
        {"has_face": 0, "category": "shapewear", "video_path": str(video_path)},
        "这是一段普通口播。",
        [],
    )
    visual_findings = [finding for finding in result.findings if finding.get("source") == "vision"]
    assert len(visual_findings) == 1
    assert result.summary["visual_count"] == 1
    assert visual_findings[0]["evidence"][0]["frame_ms"] == 4000


def test_adjacent_visual_events_are_merged(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(visual_module, "VISUAL_ENABLED", True)
    monkeypatch.setattr(visual_module, "probe_duration_ms", lambda _: 20000)
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"frame")
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr(
        visual_module,
        "_extract_frames",
        lambda video_path, duration, frame_dir: [(5000, frame_path), (12000, frame_path)],
    )
    response = {
        "events": [
            {
                "rule_id": "VIS-BODY-COMPARISON",
                "title": "穿前穿后对比",
                "category": "夸大功效/虚假塑身效果",
                "severity": "blocking",
                "frame_index": 0,
                "description": "第一帧展示穿前腹部。",
                "reason": "形成对比。",
                "suggestion": "删除对比。",
                "confidence": 0.8,
            },
            {
                "rule_id": "VIS-BODY-COMPARISON",
                "title": "穿前穿后对比",
                "category": "夸大功效/虚假塑身效果",
                "severity": "blocking",
                "frame_index": 1,
                "description": "第二帧展示穿后效果。",
                "reason": "形成对比。",
                "suggestion": "删除对比。",
                "confidence": 0.9,
            },
        ]
    }
    monkeypatch.setattr(
        visual_module,
        "_call_visual",
        lambda api_key, payload: json.dumps(response, ensure_ascii=False),
    )
    events = visual_module.review_visual(video_path)
    assert len(events) == 1
    assert len(events[0].evidence) == 2


def test_own_brand_mention_is_not_brand_confusion():
    assert visual_module._is_own_brand_description("字幕提及北极绒品牌名")
    assert not visual_module._is_own_brand_description("未经授权使用第三方品牌 logo")
