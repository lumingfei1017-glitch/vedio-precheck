from __future__ import annotations

from pathlib import Path

from app.shots import Shot, build_manifest, classify_shot, route_for_span
from app.srt import Segment


def test_classify_shot_by_face_ratio():
    assert classify_shot(0, None, None) == "broll"
    assert classify_shot(20, None, None) == "unknown"
    assert classify_shot(20, 0.9, 0.10) == "talking_closeup"
    assert classify_shot(20, 0.9, 0.02) == "talking_full"
    assert classify_shot(20, 0.3, 0.0) == "talking_side"
    assert classify_shot(20, 0.0, 0.0) == "voiceover"


def test_route_follows_visibility():
    assert Shot(0, 2000, "有口播", face_hit=0.9, face_ratio=0.1).route == "wan"
    assert Shot(0, 2000, "侧身", face_hit=0.3, face_ratio=0.0).route == "wan"
    assert Shot(0, 2000, "", face_hit=0.0, face_ratio=0.0).route == "keep"
    assert Shot(0, 2000, "有口播", face_hit=None, face_ratio=None).route == "manual"


def test_shot_shorter_than_wan_floor_uses_tts():
    shot = Shot(0, 1500, "一句话", face_hit=1.0, face_ratio=0.1)
    assert shot.route == "tts"
    assert "短于 Wan 最短" in shot.route_reason


def test_partial_violation_coverage_keeps_picture_with_tts():
    partial = Shot(0, 4000, "一整句口播", face_hit=1.0, face_ratio=0.1, coverage=0.2)
    full = Shot(0, 4000, "一整句口播", face_hit=1.0, face_ratio=0.1, coverage=0.8)
    assert partial.route == "tts"
    assert "局部补音" in partial.route_reason
    assert full.route == "wan"


def test_route_for_span_computes_coverage():
    shot = Shot(0, 4000, "一整句口播", face_hit=1.0, face_ratio=0.1)
    assert route_for_span(shot, 0, 1000).coverage == 0.25
    assert route_for_span(shot, 0, 1000).route == "tts"
    assert route_for_span(shot, 0, 3500).coverage == 0.875
    assert route_for_span(shot, 0, 3500).route == "wan"


def test_voiceover_and_no_speech_routes():
    assert Shot(0, 4000, "画外音", face_hit=0.0, face_ratio=0.0).route == "tts"
    assert Shot(0, 4000, "", face_hit=0.0, face_ratio=0.0).route == "keep"


def test_build_manifest_assigns_text_and_type(tmp_path: Path):
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"x")
    segments = [
        Segment(index=1, start_ms=0, end_ms=2000, text="第一句口播，"),
        Segment(index=2, start_ms=2600, end_ms=4000, text="第二句。"),
    ]
    shots = build_manifest(
        video,
        segments,
        duration_ms=6200,
        cuts_ms=[0, 2500, 4200],
        face_probe=lambda _video, start, _end: (0.9, 0.08) if start == 0 else (0.0, 0.0),
    )
    assert [shot.start_ms for shot in shots] == [0, 2500, 4200]
    assert shots[0].kind == "talking_closeup"
    assert shots[0].route == "wan"
    assert "第一句口播" in shots[0].text
    assert shots[1].text == "第二句。"
    assert shots[1].route == "tts"
    assert shots[2].text == ""
    assert shots[2].kind == "broll"
