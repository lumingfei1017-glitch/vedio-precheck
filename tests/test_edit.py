from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.edit import (
    _align_replace_ops_to_shots,
    _build_narration,
    _absorb_short_replace_remnants,
    _cap_oversize_cuts,
    _kept_intervals,
    _map_replace_ops_after_cuts,
    _merged_short_span,
    _merge_cut_ops,
    _merge_replace_ops,
    _resolve_cut_conflicts,
    _split_cut_ops_by_evidence,
    _expand_replace_op_by_evidence,
    _split_text_by_duration,
    _snap_replace_ops_to_shots,
    build_edit_plan,
)
from app.main import app
from app.shots import Shot
from app.srt import Segment


A_SRT = """1
00:00:00,000 --> 00:00:03,000
姐妹们，上身显瘦十几斤。
"""

B_SRT = """1
00:00:00,000 --> 00:00:03,000
姐妹们，看这条塑身衣。

2
00:00:03,000 --> 00:00:06,000
穿前小肚子，穿后曲线明显。
"""


def _create_material(client: TestClient, srt: str, has_face: str = "false") -> str:
    batch = client.post("/api/batches", json={"name": "修改方案批次"}).json()
    upload = client.post(
        f"/api/batches/{batch['id']}/materials",
        files={
            "video": ("demo.mp4", b"fake video bytes", "video/mp4"),
            "srt": ("demo.srt", srt.encode("utf-8"), "text/plain"),
        },
        data={"has_face": has_face},
    )
    return upload.json()["id"]


def test_build_a_class_plan(monkeypatch):
    monkeypatch.setattr("app.edit.TTS_ROUTE_ENABLED", True)
    with TestClient(app) as client:
        material_id = _create_material(client, A_SRT)
        client.post(f"/api/materials/{material_id}/review")
        plan = build_edit_plan(material_id)
        assert plan is not None
        assert plan["mode"] == "segment_replace"
        assert plan["operations"]
        assert plan["operations"][0]["op"] == "replace_script"


def test_build_b_class_plan(monkeypatch):
    monkeypatch.setattr("app.edit.AUTO_CUT_ENABLED", True)
    with TestClient(app) as client:
        material_id = _create_material(client, B_SRT)
        client.post(f"/api/materials/{material_id}/review")
        plan = build_edit_plan(material_id)
        assert plan is not None
        assert plan["mode"] in {"B", "mixed"}
        assert any(item["op"] == "cut" for item in plan["operations"])
        assert plan["cut_intervals"]


def test_kept_intervals():
    assert _kept_intervals(10000, [(3000, 5000)]) == [(0, 3000), (5000, 10000)]
    assert _kept_intervals(10000, []) == [(0, 10000)]


def test_multi_segment_replacement_skips_all_original_lines():
    segments = [
        Segment(1, 0, 3000, "上身显瘦十几斤"),
        Segment(2, 3000, 6000, "正面看是蚂蚁腰"),
    ]
    narration = _build_narration(
        segments,
        [{"start_ms": 0, "end_ms": 6000, "text": "这件上身包裹感更明显。"}],
    )
    assert "这件上身包裹感更明显" in narration
    assert "显瘦十几斤" not in narration
    assert "蚂蚁腰" not in narration


def test_replace_ops_are_mapped_after_cut_and_short_remnants_dropped():
    cuts = [(36619, 47080)]
    ops = [
        {"start_ms": 5600, "end_ms": 32170, "text": "第一段"},
        {"start_ms": 36170, "end_ms": 38170, "text": "被删段切开"},
        {"start_ms": 46410, "end_ms": 48330, "text": "跨删段"},
        {"start_ms": 53690, "end_ms": 55850, "text": "删段之后"},
    ]

    mapped, warnings = _map_replace_ops_after_cuts(ops, cuts, 83700)

    assert [item["text"] for item in mapped] == ["第一段", "跨删段", "删段之后"]
    assert mapped[0]["start_ms"] == 5600
    assert mapped[1] == {
        "start_ms": 36619,
        "end_ms": 37869,
        "text": "跨删段",
    }
    assert mapped[2]["start_ms"] == 43229
    assert mapped[2]["end_ms"] == 45389
    assert any("不足 0.5 秒" in warning for warning in warnings)
    assert any("重叠" in warning for warning in warnings)


def test_adjacent_replacements_in_one_shot_are_merged():
    ops = [
        {"start_ms": 79180, "end_ms": 80860, "text": "慢慢就能找到适合自己的穿着状态。", "finding_ids": ["a"]},
        {"start_ms": 81020, "end_ms": 81900, "text": "越穿越顺", "finding_ids": ["b"]},
    ]

    merged = _merge_replace_ops(ops)

    assert len(merged) == 1
    assert merged[0]["start_ms"] == 79180
    assert merged[0]["end_ms"] == 81900
    assert merged[0]["text"] == "慢慢就能找到适合自己的穿着状态，越穿越顺"
    assert merged[0]["finding_ids"] == ["a", "b"]


def test_replacements_are_expanded_to_shot_boundaries(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(edit_module, "_load_video_path", lambda _: str(video))
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])
    monkeypatch.setattr(
        "app.shots.cached_manifest",
        lambda *args, **kwargs: [
            Shot(start_ms=78200, end_ms=83700, text="坚持穿下去，你的身材能塑形定型，越穿越有型。")
        ],
    )
    ops = [
        {"start_ms": 79180, "end_ms": 80860, "text": "第一句", "finding_ids": ["a"]},
        {"start_ms": 81020, "end_ms": 81900, "text": "第二句", "finding_ids": ["b"]},
    ]

    snapped = _snap_replace_ops_to_shots(
        "m1",
        {"duration_ms": 83700},
        ops,
    )

    assert [item["start_ms"] for item in snapped] == [78200, 78200]
    assert [item["end_ms"] for item in snapped] == [83700, 83700]


def test_too_short_replace_remnant_is_absorbed_into_cut():
    replace_ops = [
        {
            "start_ms": 46410,
            "end_ms": 48330,
            "text": "穿上之后整个型看着更饱满，视觉上会精神一些。",
            "finding_ids": ["replace-1"],
            "suggestion_id": "s1",
        }
    ]
    cut_ops = [
        {
            "start_ms": 36619,
            "end_ms": 47080,
            "instruction": "删除穿前穿后对比",
            "finding_ids": ["cut-1"],
            "suggestion_id": "s2",
        }
    ]

    remaining, cuts, warnings = _absorb_short_replace_remnants(
        replace_ops,
        cut_ops,
        83700,
    )

    assert remaining == []
    assert cuts[0]["start_ms"] == 36619
    assert cuts[0]["end_ms"] == 48330
    assert cuts[0]["finding_ids"] == ["cut-1"]
    assert "穿前穿后对比" in cuts[0]["instruction"]
    assert any("已并入删段" in warning for warning in warnings)


def test_small_cross_shot_overlap_still_snaps_to_dominant_shot(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(edit_module, "_load_video_path", lambda _: str(video))
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])
    monkeypatch.setattr(
        "app.shots.cached_manifest",
        lambda *args, **kwargs: [
            Shot(start_ms=52800, end_ms=55800, text="你把它穿上身直接显瘦十几斤"),
            Shot(start_ms=55800, end_ms=60000, text="它的一个承压能力呢会更强"),
        ],
    )
    ops = [
        {
            "start_ms": 53690,
            "end_ms": 55850,
            "text": "你把它穿上身，整体看着就更紧致利落。",
            "finding_ids": ["a"],
        }
    ]

    snapped = _snap_replace_ops_to_shots("m1", {"duration_ms": 60000}, ops)

    assert snapped[0]["start_ms"] == 52800
    assert snapped[0]["end_ms"] == 55800


def test_small_tail_of_long_shot_is_not_expanded(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(edit_module, "_load_video_path", lambda _: str(video))
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])
    monkeypatch.setattr(
        "app.shots.cached_manifest",
        lambda *args, **kwargs: [
            Shot(start_ms=31233, end_ms=38167, text="长镜头前段"),
            Shot(start_ms=38167, end_ms=40967, text="长镜头后段"),
        ],
    )
    ops = [
        {
            "start_ms": 36170,
            "end_ms": 38170,
            "text": "只改镜头尾部",
            "finding_ids": ["a"],
        }
    ]

    snapped = _snap_replace_ops_to_shots("m1", {"duration_ms": 40967}, ops)

    assert snapped[0]["start_ms"] == 36170
    assert snapped[0]["end_ms"] == 38170


def test_semantic_evidence_clusters_avoid_long_empty_replacement():
    finding = {
        "evidence": [
            {"start_ms": 5600, "end_ms": 7000},
            {"start_ms": 7160, "end_ms": 8360},
            {"start_ms": 8480, "end_ms": 11360},
            {"start_ms": 11440, "end_ms": 12520},
            {"start_ms": 12600, "end_ms": 14200},
            {"start_ms": 30090, "end_ms": 32170},
        ]
    }
    op = {
        "start_ms": 5600,
        "end_ms": 32170,
        "text": "它整个半身是这种高弹分压的设计，腰腹、腰侧这些位置包得比较服帖，正面侧面看着都挺利落。",
        "finding_ids": ["f1"],
        "suggestion_id": "s1",
    }

    expanded = _expand_replace_op_by_evidence(op, finding)

    assert [(item["start_ms"], item["end_ms"]) for item in expanded] == [
        (5600, 14200),
        (30090, 32170),
    ]
    assert "高弹分压" in expanded[0]["text"]
    assert expanded[1]["text"] == "正面侧面看着都挺利落。"


def test_text_split_respects_cluster_duration_share():
    chunks = _split_text_by_duration(
        "它整个半身是这种高弹分压的设计，腰腹、腰侧这些位置包得比较服帖，正面侧面看着都挺利落。",
        [8600, 2080],
        chars_per_second=5.5,
    )

    assert len(chunks) == 2
    assert len(chunks[0]) > len(chunks[1])
    assert chunks[1] == "正面侧面看着都挺利落。"


def test_cross_shot_replacement_is_split_per_shot(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(edit_module, "_load_video_path", lambda _: str(video))
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])
    monkeypatch.setattr(
        "app.shots.cached_manifest",
        lambda *args, **kwargs: [
            Shot(0, 2000, "前半句口播", face_hit=1.0, face_ratio=0.1),
            Shot(2000, 5000, "后半句口播", face_hit=1.0, face_ratio=0.1),
        ],
    )
    ops = [
        {
            "start_ms": 500,
            "end_ms": 4500,
            "text": "穿上之后腰腹包裹感更明显，整体线条看着更利落一些。",
            "finding_ids": ["a"],
        }
    ]

    aligned, manual, warnings = _align_replace_ops_to_shots(
        "m1",
        {"duration_ms": 5000},
        ops,
    )

    assert manual == []
    assert [(item["start_ms"], item["end_ms"]) for item in aligned] == [
        (0, 2000),
        (2000, 5000),
    ]
    assert [item["route"] for item in aligned] == ["wan", "wan"]
    assert [item["finding_ids"] for item in aligned] == [["a"], ["a"]]
    assert "".join(item["text"] for item in aligned) == ops[0]["text"]
    assert any("已按镜头拆成 2 段" in warning for warning in warnings)


def test_cross_shot_replacement_with_unusable_text_goes_manual(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(edit_module, "_load_video_path", lambda _: str(video))
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])
    monkeypatch.setattr(
        "app.shots.cached_manifest",
        lambda *args, **kwargs: [
            Shot(0, 2000, "前半句口播", face_hit=1.0, face_ratio=0.1),
            Shot(2000, 5000, "后半句口播", face_hit=1.0, face_ratio=0.1),
        ],
    )
    ops = [{"start_ms": 500, "end_ms": 4500, "text": "看", "finding_ids": ["a"]}]

    aligned, manual, warnings = _align_replace_ops_to_shots(
        "m1",
        {"duration_ms": 5000},
        ops,
    )

    assert aligned == []
    assert len(manual) == 1
    assert "跨 2 个镜头" in manual[0]["reason"]
    assert any("文案拆不开" in warning for warning in warnings)


def test_fast_cut_cross_shot_stays_one_tts_unit(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(edit_module, "_load_video_path", lambda _: str(video))
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])
    monkeypatch.setattr(
        "app.shots.cached_manifest",
        lambda *args, **kwargs: [
            Shot(0, 3000, "150的穿强压", face_hit=1.0, face_ratio=0.1),
            Shot(3000, 4000, "120的专穿北极绒", face_hit=1.0, face_ratio=0.1),
            Shot(4000, 4500, "90斤的还是穿北极绒", face_hit=1.0, face_ratio=0.1),
        ],
    )
    ops = [
        {
            "start_ms": 0,
            "end_ms": 4500,
            "text": "150的穿强压，120的专穿北极绒，90斤的还是穿北极绒。",
            "finding_ids": ["a"],
        }
    ]

    aligned, manual, warnings = _align_replace_ops_to_shots(
        "m1",
        {"duration_ms": 4500},
        ops,
    )

    assert manual == []
    assert len(aligned) == 1
    assert aligned[0]["route"] == "tts"
    assert aligned[0]["start_ms"] == 0
    assert aligned[0]["end_ms"] == 4500
    assert aligned[0]["cross_shots"] == 3
    assert aligned[0]["text"] == ops[0]["text"]
    assert any("整段只换音轨" in warning for warning in warnings)


def test_merge_short_span_pads_to_wan_minimum():
    shots = [
        Shot(0, 1000, "第一句口播", face_hit=1.0, face_ratio=0.1),
        Shot(1000, 2600, "第二句口播", face_hit=1.0, face_ratio=0.1),
    ]

    merged = _merged_short_span(
        shots,
        0,
        1000,
        boundary_ok=lambda _prev, _next: True,
        min_seconds=2.0,
        max_seconds=4.0,
    )

    assert merged is not None
    assert merged[0] == 0
    assert merged[1] == 2600
    assert len(merged[2]) == 2


def test_merge_short_span_refuses_real_scene_change():
    shots = [
        Shot(0, 1000, "第一句口播", face_hit=1.0, face_ratio=0.1),
        Shot(1000, 2600, "第二句口播", face_hit=1.0, face_ratio=0.1),
    ]

    merged = _merged_short_span(
        shots,
        0,
        1000,
        boundary_ok=lambda _prev, _next: False,
        min_seconds=2.0,
        max_seconds=4.0,
    )

    assert merged is None


def test_short_shot_is_merged_into_one_wan_unit(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(edit_module, "_load_video_path", lambda _: str(video))
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])
    monkeypatch.setattr(
        "app.shots.cached_manifest",
        lambda *args, **kwargs: [
            Shot(0, 1000, "穿前小肚子", face_hit=1.0, face_ratio=0.1),
            Shot(1000, 2600, "穿后看着更顺", face_hit=1.0, face_ratio=0.1),
        ],
    )
    monkeypatch.setattr("app.shots.can_merge_across", lambda *_args, **_kwargs: True)
    ops = [
        {
            "start_ms": 0,
            "end_ms": 1000,
            "text": "上身版型看着更利落。",
            "finding_ids": ["a"],
        }
    ]

    aligned, manual, warnings = _align_replace_ops_to_shots(
        "m1",
        {"duration_ms": 2600},
        ops,
    )

    assert manual == []
    assert len(aligned) == 1
    assert aligned[0]["route"] == "wan"
    assert aligned[0]["start_ms"] == 0
    assert aligned[0]["end_ms"] == 2600
    assert aligned[0]["merged_shots"] == 2
    assert any("已合并 2 个相邻镜头" in warning for warning in warnings)


def test_merged_unit_with_overlong_text_falls_back_to_tts(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(edit_module, "_load_video_path", lambda _: str(video))
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])
    monkeypatch.setattr(
        "app.shots.cached_manifest",
        lambda *args, **kwargs: [
            Shot(0, 1000, "穿前小肚子", face_hit=1.0, face_ratio=0.1),
            Shot(1000, 2600, "穿后看着更顺", face_hit=1.0, face_ratio=0.1),
        ],
    )
    monkeypatch.setattr("app.shots.can_merge_across", lambda *_args, **_kwargs: True)
    ops = [
        {
            "start_ms": 0,
            "end_ms": 1000,
            "text": "这件上身之后腰腹位置的包裹感会更明显一些，平时穿搭也更好看。",
            "finding_ids": ["a"],
        }
    ]

    aligned, manual, warnings = _align_replace_ops_to_shots(
        "m1",
        {"duration_ms": 2600},
        ops,
    )

    assert manual == []
    assert aligned[0]["route"] == "tts"
    assert any("超出上限" in warning for warning in warnings)


def test_long_semantic_cut_is_split_by_evidence_clusters():
    op = {
        "start_ms": 39830,
        "end_ms": 68080,
        "finding_ids": ["f1"],
        "instruction": "删除确定性效果承诺",
        "evidence": [
            {"start_ms": 39830, "end_ms": 42670},
            {"start_ms": 42750, "end_ms": 43950},
            {"start_ms": 49880, "end_ms": 52920},
            {"start_ms": 53480, "end_ms": 55480},
            {"start_ms": 61640, "end_ms": 63000},
            {"start_ms": 66560, "end_ms": 68080},
        ],
    }

    split = _split_cut_ops_by_evidence([op])

    assert [(item["start_ms"], item["end_ms"]) for item in split] == [
        (39830, 43950),
        (49880, 55480),
        (61640, 63000),
        (66560, 68080),
    ]
    assert all("按证据簇" in item["instruction"] for item in split)
    assert all(item["finding_ids"] == ["f1"] for item in split)


def test_merge_cut_ops_refuses_to_chain_beyond_limit():
    ops = [
        {"start_ms": 39346, "end_ms": 44966, "finding_ids": ["a"], "instruction": "A"},
        {"start_ms": 39830, "end_ms": 68080, "finding_ids": ["b"], "instruction": "B"},
        {"start_ms": 67450, "end_ms": 73070, "finding_ids": ["c"], "instruction": "C"},
    ]

    merged = _merge_cut_ops(ops, max_span_ms=12000)

    assert len(merged) == 3


def test_oversize_cut_is_downgraded_to_manual():
    ops = [
        {"start_ms": 0, "end_ms": 9000, "finding_ids": ["a"], "instruction": "短"},
        {"start_ms": 20000, "end_ms": 50000, "finding_ids": ["b"], "instruction": "长"},
    ]

    kept, manual = _cap_oversize_cuts(ops, 90000)

    assert [item["end_ms"] for item in kept] == [9000]
    assert len(manual) == 1
    assert "超过自动上限" in manual[0]["reason"]


def test_high_confidence_visual_beats_weak_rewrite():
    replace_ops = [
        {"start_ms": 10000, "end_ms": 14000, "text": "改写", "confidence": 0.8}
    ]
    cut_ops = [
        {"start_ms": 9000, "end_ms": 15000, "confidence": 0.95, "finding_ids": ["v"]}
    ]

    _replace, cuts, warnings = _resolve_cut_conflicts(replace_ops, cut_ops)

    assert [(item["start_ms"], item["end_ms"]) for item in cuts] == [(9000, 15000)]
    assert any("删段优先" in warning for warning in warnings)


def test_borderline_visual_yields_to_confident_rewrite():
    replace_ops = [
        {"start_ms": 10000, "end_ms": 14000, "text": "改写", "confidence": 0.9}
    ]
    cut_ops = [
        {"start_ms": 9000, "end_ms": 15000, "confidence": 0.82, "finding_ids": ["v"]}
    ]

    _replace, cuts, warnings = _resolve_cut_conflicts(replace_ops, cut_ops)

    assert [(item["start_ms"], item["end_ms"]) for item in cuts] == [
        (9000, 10000),
        (14000, 15000),
    ]
    assert any("改写优先" in warning for warning in warnings)


def test_very_confident_rewrite_removes_overlapping_cut():
    replace_ops = [
        {"start_ms": 9000, "end_ms": 15000, "text": "改写", "confidence": 0.97}
    ]
    cut_ops = [
        {"start_ms": 9500, "end_ms": 14000, "confidence": 0.95, "finding_ids": ["v"]}
    ]

    _replace, cuts, _warnings = _resolve_cut_conflicts(replace_ops, cut_ops)

    assert cuts == []




def test_default_policy_sends_cut_and_tts_to_manual(monkeypatch):
    monkeypatch.setattr("app.edit.TTS_ROUTE_ENABLED", False)
    monkeypatch.setattr("app.edit.AUTO_CUT_ENABLED", False)
    with TestClient(app) as client:
        material_id = _create_material(client, B_SRT)
        client.post(f"/api/materials/{material_id}/review")
        plan = build_edit_plan(material_id)

    assert plan is not None
    assert plan["cut_intervals"] == []
    assert plan["mode"] in {"manual", "none"}
    assert any(item["op"] == "manual" for item in plan["operations"])


def test_worklist_exports_markdown_and_srt(monkeypatch):
    from app.worklist import build_worklist

    monkeypatch.setattr("app.edit.AUTO_CUT_ENABLED", False)
    with TestClient(app) as client:
        material_id = _create_material(client, B_SRT)
        client.post(f"/api/materials/{material_id}/review")
        worklist = build_worklist(material_id)

    assert worklist is not None
    assert "需人工处理" in worklist["markdown"]
    assert worklist["manual_count"] >= 1
    assert "-->" in worklist["srt"]


def _wan_op(start, end, text):
    return {"start_ms": start, "end_ms": end, "text": text, "route": "wan"}


def test_wan_replace_applies_successes_and_skips_failures(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"original")
    out = tmp_path / "out.mp4"
    srt = tmp_path / "out.srt"
    calls = []

    def fake_replace(video_path, start, end, text, stage, work_dir=None):
        calls.append(text)
        if text.startswith("失败"):
            raise RuntimeError("Wan 输出超时长")
        Path(stage).write_bytes(b"stage-" + text.encode())
        return {"padded_ms": 0}

    monkeypatch.setattr("app.replace.replace_shot", fake_replace)
    monkeypatch.setattr(edit_module, "probe_duration_ms", lambda _: 60000)
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])

    result = edit_module._execute_wan_replace(
        "m1",
        video,
        [
            _wan_op(0, 3000, "改后文案甲乙丙丁戊己庚辛"),
            _wan_op(5000, 8000, "失败文案甲乙丙丁戊己庚辛"),
            _wan_op(10000, 13000, "改后文案二三四五六七八九十甲乙"),
        ],
        srt,
        out,
        tmp_path,
        None,
        {"warnings": []},
    )

    assert [item["text"] for item in result["applied"]] == [
        "改后文案甲乙丙丁戊己庚辛",
        "改后文案二三四五六七八九十甲乙",
    ]
    assert len(result["skipped"]) == 1
    assert "超时长" in result["skipped"][0]["reason"]
    assert out.read_bytes() == "stage-改后文案二三四五六七八九十甲乙".encode()
    assert result["mode"] == "wan_replace"
    assert calls == [
        "改后文案甲乙丙丁戊己庚辛",
        "失败文案甲乙丙丁戊己庚辛",
        "改后文案二三四五六七八九十甲乙",
    ]


def test_wan_replace_all_failed_keeps_original(monkeypatch, tmp_path):
    import app.edit as edit_module

    video = tmp_path / "video.mp4"
    video.write_bytes(b"original")
    out = tmp_path / "out.mp4"

    def fake_replace(*args, **kwargs):
        raise RuntimeError("文案超出容量")

    monkeypatch.setattr("app.replace.replace_shot", fake_replace)
    monkeypatch.setattr(edit_module, "probe_duration_ms", lambda _: 60000)
    monkeypatch.setattr(edit_module, "_load_segments", lambda _: [])

    result = edit_module._execute_wan_replace(
        "m1",
        video,
        [_wan_op(0, 3000, "改后A")],
        tmp_path / "out.srt",
        out,
        tmp_path,
        None,
        {"warnings": []},
    )

    assert result["applied"] == []
    assert result["mode"] == "unchanged"
    assert out.read_bytes() == b"original"
    assert any("保留原样" in warning for warning in result["warnings"])


def test_wan_preflight_enforces_wan_rules():
    import app.edit as edit_module

    ok = {"start_ms": 0, "end_ms": 4000, "text": "这件上身包裹感挺明显，腰腹看着更利落。"}
    assert edit_module._wan_preflight(ok, "这件上身显瘦十几斤") is None

    short = {"start_ms": 0, "end_ms": 1500, "text": "这件版型挺利落。"}
    assert "短于 Wan 最短" in edit_module._wan_preflight(short, "短镜头")

    long_text = {"start_ms": 0, "end_ms": 4000, "text": "这件上身包裹感挺明显腰腹看着更利落日常穿搭也更服帖"}
    assert "容量" in edit_module._wan_preflight(long_text, "原句")

    repeat = {"start_ms": 0, "end_ms": 4000, "text": "这件上身显瘦但是腰腹包裹感更明显一些"}
    assert "复读" in edit_module._wan_preflight(repeat, "这件上身显瘦十几斤")
