from __future__ import annotations

import pytest

from app.replace import ReplaceError, build_splice_graph, plan_replace
from app.rewrite import RewriteResult, _off_budget


def test_splice_graph_pads_short_generated_clip_to_slot():
    graph, pad, stretch = build_splice_graph(
        width=1080,
        height=1920,
        fps="30/1",
        start=10.0,
        end=12.4,
        generated_ms=2000,
    )

    assert pad == pytest.approx(0.4, abs=1e-6)
    assert stretch is None
    assert "tpad=stop_mode=clone:stop_duration=0.400" in graph
    assert "apad=pad_dur=0.400" in graph
    assert "trim=0:2.400" in graph
    assert "atrim=0:2.400" in graph


def test_splice_graph_keeps_slot_length_when_exact():
    graph, pad, stretch = build_splice_graph(
        width=1080,
        height=1920,
        fps="30/1",
        start=0.0,
        end=3.0,
        generated_ms=3000,
    )

    assert pad == 0
    assert stretch is None
    assert "tpad" not in graph
    assert "apad" not in graph
    assert "trim=0:3.000" in graph


def test_splice_graph_stretches_small_overshoot_instead_of_trimming():
    graph, pad, stretch = build_splice_graph(
        width=720,
        height=1280,
        fps="25/1",
        start=4.0,
        end=7.0,
        generated_ms=3060,
    )

    assert pad == 0
    assert stretch == pytest.approx(1.02, abs=1e-6)
    assert "atempo=1.020000" in graph
    assert "setpts=(PTS-STARTPTS)*0.980392" in graph
    assert "trim=0:3.000" in graph


def test_splice_graph_mixes_original_background_when_available():
    graph, _pad, _stretch = build_splice_graph(
        width=720,
        height=1280,
        fps="30/1",
        start=10.0,
        end=12.4,
        generated_ms=2000,
        background=True,
    )

    assert "[2:a]atrim=10.000:12.400" in graph
    assert "amix=inputs=2:duration=first:normalize=0" in graph


def test_splice_graph_without_background_passes_voice_through():
    graph, _pad, _stretch = build_splice_graph(
        width=720,
        height=1280,
        fps="30/1",
        start=10.0,
        end=12.4,
        generated_ms=2000,
        background=False,
    )

    assert "amix" not in graph
    assert "[voice]anull[a1]" in graph


def test_wan_char_budget_follows_integer_generation_duration():
    plan = plan_replace(0.0, 5.5, "合规口播文案")

    assert plan.duration == 5
    assert plan.budget == 25


def test_overlong_script_is_rejected_before_generation():
    text = "这件塑身衣上身之后包裹感更明显腰腹位置看着更利落日常穿搭也更服帖舒服很多姐妹都在回购"

    with pytest.raises(ReplaceError) as excinfo:
        plan_replace(82.6, 88.1, text)

    assert "吞字糊音" in str(excinfo.value)
    assert "精简" in str(excinfo.value)


def test_script_within_capacity_passes():
    plan = plan_replace(82.6, 88.1, "这件上身包裹感挺明显，腰腹看着更利落。")

    assert plan.duration == 5
    assert plan.chars <= 29


def test_rewrite_over_budget_detection():
    long = RewriteResult(
        action="replace_script",
        after_text="很长的一段口播",
        edit_instruction="",
        kept_selling_point="",
        estimated_duration_ms=7300,
        alternatives=(),
        confidence=0.8,
    )
    short = RewriteResult(
        action="replace_script",
        after_text="短一点",
        edit_instruction="",
        kept_selling_point="",
        estimated_duration_ms=5200,
        alternatives=(),
        confidence=0.8,
    )
    suggestion = {"original_duration_ms": 5500}

    too_short = RewriteResult(
        action="replace_script",
        after_text="太短",
        edit_instruction="",
        kept_selling_point="",
        estimated_duration_ms=2600,
        alternatives=(),
        confidence=0.8,
    )

    assert _off_budget(long, suggestion) == "over"
    assert _off_budget(short, suggestion) is None
    assert _off_budget(too_short, suggestion) == "under"
