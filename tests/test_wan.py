from __future__ import annotations

import pytest

from app.replace import ReplaceError, validate_generated_duration
from app.wan import WanError, build_shot_prompt, char_budget, planned_duration, text_fit


def test_duration_follows_reference_and_never_exceeds_by_much():
    assert planned_duration(5.5) == 5
    # 6.94 → 7 只超 0.86%，D 版实测同样幅度清晰可用
    assert planned_duration(6.94) == 7
    assert planned_duration(2.0) == 2
    assert planned_duration(45.0) == 30


def test_duration_rounds_up_only_when_tail_is_large():
    # 4.92s floor 成 4 会白丢 0.92 秒容量，向上取整只超 1.6%
    assert planned_duration(4.92) == 5
    # 5.52s 向上取整要超 8.7%，超过 6% 上限，仍取 floor
    assert planned_duration(5.52) == 5


def test_char_budget_matches_speech_rate():
    assert char_budget(5.5) == 27
    assert char_budget(6.1) == 30


def test_text_fit_flags_filler_and_rush():
    assert text_fit(20, 5) == "short"   # H 版：空档被原句碎片填空
    assert text_fit(28, 5) == "ok"      # I / J 版
    assert text_fit(28, 7) == "short"   # G 版：同样的字塞进 7 秒，发糊
    assert text_fit(60, 5) == "long"


def test_prompt_carries_dialogue_and_constraints():
    prompt = build_shot_prompt("这款平时穿也合适。")
    assert '"这款平时穿也合适。"' in prompt
    assert "Lip sync" in prompt
    assert "禁止新增任何文字" in prompt
    assert "prompt" not in prompt.lower()


def test_empty_line_is_rejected():
    with pytest.raises(WanError):
        build_shot_prompt("   ")


def test_wan_output_longer_than_reference_is_rejected():
    validate_generated_duration(5500, 5500)
    with pytest.raises(ReplaceError, match="超过参考镜头"):
        validate_generated_duration(7000, 5500)
