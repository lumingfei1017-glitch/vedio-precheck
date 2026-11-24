from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

import app.tts as tts
from app.asr import probe_duration_ms
from app.config import FFMPEG_PATH
from app.edit import (
    extract_stable_voice_reference,
    extract_voice_reference,
    replace_audio_interval,
)


def test_synthesize_marks_edge_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "TTS_PROVIDER", "auto")
    monkeypatch.setattr(tts, "VOICE_CLONE_ENABLED", True)
    monkeypatch.setattr(tts, "TTS_VOICE_ID", "voice-test")
    monkeypatch.setattr(
        tts,
        "_cosyvoice_synthesize",
        lambda *args, **kwargs: (_ for _ in ()).throw(tts.TTSError("no permission")),
    )

    def edge(text: str, output: Path) -> None:
        output.write_bytes(b"fake-mp3" * 300)

    monkeypatch.setattr(tts, "_edge_synthesize", edge)
    monkeypatch.setattr(tts, "probe_duration_ms", lambda _: 1200)

    result = tts.synthesize("合规口播", tmp_path / "edge.mp3")

    assert result.provider == "edge"
    assert result.voice_source == "edge_fallback"
    assert result.voice_id is None
    assert any("已回退 Edge" in warning for warning in result.warnings)


def test_synthesize_reports_cloned_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "TTS_PROVIDER", "auto")
    monkeypatch.setattr(tts, "VOICE_CLONE_ENABLED", True)
    monkeypatch.setattr(tts, "probe_duration_ms", lambda _: 900)

    def clone(text: str, voice_id: str, output: Path) -> None:
        assert voice_id == "cloned-voice"
        output.write_bytes(b"clone" * 300)

    monkeypatch.setattr(tts, "_cosyvoice_synthesize", clone)
    result = tts.synthesize(
        "合规口播",
        tmp_path / "clone.mp3",
        voice_id="cloned-voice",
    )

    assert result.provider == "cosyvoice"
    assert result.voice_source == "cloned"
    assert result.voice_id == "cloned-voice"


def test_explicit_cosyvoice_provider_does_not_silently_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "TTS_PROVIDER", "cosyvoice")
    monkeypatch.setattr(tts, "VOICE_CLONE_ENABLED", True)
    monkeypatch.setattr(tts, "TTS_VOICE_ID", "voice-test")
    monkeypatch.setattr(
        tts,
        "_cosyvoice_synthesize",
        lambda *args, **kwargs: (_ for _ in ()).throw(tts.TTSError("no permission")),
    )

    import pytest

    with pytest.raises(tts.TTSError, match="克隆 TTS 不可用"):
        tts.synthesize("合规口播", tmp_path / "must-fail.mp3")


def test_voice_id_is_cached_by_reference_hash(monkeypatch, tmp_path):
    calls = {"create": 0}

    class Voices:
        @staticmethod
        def create(**kwargs):
            calls["create"] += 1
            return types.SimpleNamespace(voice_id="cached-voice")

        @staticmethod
        def get(_voice_id):
            return types.SimpleNamespace(status="OK")

    class Client:
        def __init__(self, **kwargs):
            self.audio = types.SimpleNamespace(voices=Voices())

    monkeypatch.setitem(sys.modules, "dashscope", types.SimpleNamespace(Client=Client))
    monkeypatch.setattr(tts, "VOICE_CACHE_DIR", tmp_path / "voices")
    monkeypatch.setattr(
        tts,
        "_convert_to_wav",
        lambda source, target: target.write_bytes(b"wav" * 300),
    )
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-audio")

    first = tts._voice_id_for_reference(reference)
    second = tts._voice_id_for_reference(reference)

    assert first == second == "cached-voice"
    assert calls["create"] == 1


def _run_ffmpeg(*args: str) -> None:
    assert FFMPEG_PATH is not None
    import subprocess

    completed = subprocess.run(
        [str(FFMPEG_PATH), "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _dominant_frequency(audio: Path, start: float, end: float) -> int:
    import subprocess

    assert FFMPEG_PATH is not None
    completed = subprocess.run(
        [
            str(FFMPEG_PATH),
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(audio),
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    samples = np.frombuffer(completed.stdout, dtype=np.float32)
    assert samples.size > 100, completed.stderr[-300:]
    window = np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(samples * window))
    frequencies = np.fft.rfftfreq(samples.size, d=1 / 8000)
    return int(round(float(frequencies[int(np.argmax(spectrum))])))


def test_segment_replace_keeps_audio_outside_target(tmp_path):
    source = tmp_path / "source.mp4"
    replacement = tmp_path / "tts.wav"
    background = tmp_path / "background.wav"
    output = tmp_path / "output.mp4"

    _run_ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:s=160x120:d=3",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=3",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
    )
    _run_ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-ac", "1", "-ar", "24000", str(replacement),
    )
    _run_ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=100:duration=3",
        "-ac", "1", "-ar", "24000", str(background),
    )

    replace_audio_interval(
        source,
        replacement,
        1000,
        2000,
        3000,
        output,
        background_audio=background,
    )

    assert abs(probe_duration_ms(output) - 3000) <= 200
    assert _dominant_frequency(output, 0.15, 0.65) == 1000
    assert _dominant_frequency(output, 2.25, 2.75) == 1000
    target_frequency = _dominant_frequency(output, 1.2, 1.8)
    assert target_frequency in {100, 110, 440, 450}


def test_voice_reference_prefers_region_outside_edit(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "reference.wav"
    _run_ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:s=160x120:d=30",
        "-f", "lavfi", "-i", "sine=frequency=220:duration=30",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
    )

    reference = extract_voice_reference(source, 10000, 13000, 30000, output)

    assert reference == output
    assert output.is_file()
    assert 16000 <= probe_duration_ms(output) <= 18000


def test_fit_audio_rejects_unrecoverable_overlength(monkeypatch, tmp_path):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"fake")
    monkeypatch.setattr(tts, "probe_duration_ms", lambda _: 5000)

    with pytest.raises(tts.TTSError, match="仍会截断"):
        tts.fit_audio(source, tmp_path / "out.mp3", 4000)


def test_stable_voice_reference_uses_30_second_continuous_audio(tmp_path):
    source = tmp_path / "vocals.wav"
    output = tmp_path / "stable-reference.wav"
    _run_ffmpeg(
        "-f", "lavfi", "-i", "sine=frequency=220:duration=40",
        "-ac", "1", "-ar", "16000", str(source),
    )

    result = extract_stable_voice_reference(source, output, target_ms=30000)

    assert result == output
    assert 29000 <= probe_duration_ms(output) <= 31000
