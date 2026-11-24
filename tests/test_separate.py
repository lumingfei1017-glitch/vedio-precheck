from __future__ import annotations

from pathlib import Path

import pytest

import app.separate as separate


def _fake_runner(calls: list[int]):
    def run(wav: Path, work: Path) -> tuple[Path, Path]:
        calls.append(1)
        vocals = work / "vocals.wav"
        background = work / "no_vocals.wav"
        vocals.write_bytes(b"vocals" * 200)
        background.write_bytes(b"bg" * 200)
        return vocals, background

    return run


@pytest.fixture()
def sample(tmp_path, monkeypatch):
    monkeypatch.setattr(separate, "extract_wav", lambda source, dest: dest.write_bytes(b"wav" * 300))
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video-bytes")
    return source, tmp_path / "stems"


def test_separate_caches_by_content(sample):
    source, stems_dir = sample
    calls: list[int] = []
    runner = _fake_runner(calls)

    first = separate.separate_audio(source, runner=runner, stems_dir=stems_dir, model="m", device="cpu")
    second = separate.separate_audio(source, runner=runner, stems_dir=stems_dir, model="m", device="cpu")

    assert first.cached is False
    assert second.cached is True
    assert len(calls) == 1
    assert second.vocals.is_file() and second.background.is_file()
    assert second.cache_key == first.cache_key


def test_separate_can_bypass_cache(sample):
    source, stems_dir = sample
    calls: list[int] = []
    runner = _fake_runner(calls)

    separate.separate_audio(source, runner=runner, stems_dir=stems_dir, model="m", device="cpu")
    separate.separate_audio(
        source, runner=runner, stems_dir=stems_dir, model="m", device="cpu", use_cache=False
    )

    assert len(calls) == 2


def test_separate_reports_missing_demucs(sample, monkeypatch):
    source, stems_dir = sample
    monkeypatch.setattr(separate, "DEMUCS_PYTHON", None)

    with pytest.raises(separate.SeparateError, match="DEMUCS_PYTHON"):
        separate.separate_audio(source, stems_dir=stems_dir, model="m", device="cpu")
