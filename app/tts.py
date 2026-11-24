from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .asr import probe_duration_ms
from .config import (
    DASHSCOPE_API_KEY,
    FFMPEG_PATH,
    TTS_MAX_TEMPO,
    TTS_MODEL,
    TTS_PROVIDER,
    TTS_RATE,
    TTS_SAMPLE_RATE,
    TTS_VOICE,
    TTS_VOICE_ID,
    VOICE_CACHE_DIR,
    VOICE_CLONE_ENABLED,
    VOICE_REFERENCE_PATH,
)


class TTSError(RuntimeError):
    pass


@dataclass(frozen=True)
class TTSResult:
    audio_path: Path
    provider: str
    voice_id: str | None
    voice_source: str
    duration_ms: int
    warnings: tuple[str, ...] = ()


def synthesize(
    text: str,
    output_path: Path,
    *,
    voice_id: str | None = None,
    reference_audio: Path | None = None,
) -> TTSResult:
    if not text.strip():
        raise TTSError("缺少配音文案")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    active_voice_id = (voice_id or TTS_VOICE_ID).strip() or None
    active_reference = reference_audio or (
        Path(VOICE_REFERENCE_PATH) if VOICE_REFERENCE_PATH else None
    )

    clone_requested = TTS_PROVIDER in {"auto", "cosyvoice"} and VOICE_CLONE_ENABLED
    if clone_requested and (active_voice_id or (active_reference and active_reference.is_file())):
        try:
            cloned_voice_id = active_voice_id or _voice_id_for_reference(active_reference)
            _cosyvoice_synthesize(text, cloned_voice_id, output_path)
            return TTSResult(
                audio_path=output_path,
                provider="cosyvoice",
                voice_id=cloned_voice_id,
                voice_source="cloned",
                duration_ms=probe_duration_ms(output_path),
                warnings=tuple(warnings),
            )
        except Exception as exc:  # noqa: BLE001
            if TTS_PROVIDER == "cosyvoice":
                raise TTSError(f"克隆 TTS 不可用：{str(exc)[:180]}") from exc
            else:
                warnings.append(f"克隆 TTS 不可用，已回退 Edge：{str(exc)[:180]}")

    if TTS_PROVIDER == "cosyvoice":
        raise TTSError("没有可用的克隆音色，且 TTS_PROVIDER=cosyvoice")

    _edge_synthesize(text, output_path)
    if not warnings:
        warnings.append("未使用克隆音色，当前为 Edge 测试音色")
    return TTSResult(
        audio_path=output_path,
        provider="edge",
        voice_id=None,
        voice_source="edge_fallback",
        duration_ms=probe_duration_ms(output_path),
        warnings=tuple(warnings),
    )


def fit_audio(
    source: Path,
    output: Path,
    target_duration_ms: int,
    *,
    max_tempo: float | None = None,
) -> tuple[Path, float, list[str]]:
    """Fit synthesized audio to the target segment without extreme time stretch."""
    if FFMPEG_PATH is None:
        raise TTSError("未找到 FFmpeg")
    if target_duration_ms <= 0:
        raise TTSError("目标片段时长无效")
    source_ms = probe_duration_ms(source)
    if source_ms <= 0:
        raise TTSError("TTS 音频时长无效")
    allowed = max_tempo or TTS_MAX_TEMPO
    factor = 1.0
    warnings: list[str] = []
    if source_ms > target_duration_ms * 1.02:
        required = source_ms / target_duration_ms
        if required > allowed:
            raise TTSError(
                "配音需要 "
                f"{source_ms / 1000:.2f}s，目标区间只有 {target_duration_ms / 1000:.2f}s；"
                f"即使加速到 {allowed:.2f}x 仍会截断，请缩短文案或扩大修改区间"
            )
        factor = max(0.9, required)
        warnings.append(f"文案偏长，已用 {factor:.2f}x 加速贴合目标时长")
    completed = subprocess.run(
        [
            str(FFMPEG_PATH),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-filter:a",
            f"atempo={factor:.4f},apad,atrim=0:{target_duration_ms / 1000:.3f}",
            "-ac",
            "1",
            "-ar",
            str(TTS_SAMPLE_RATE),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "192k",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        raise TTSError(f"TTS 对齐失败：{completed.stderr[-300:]}")
    return output, factor, warnings


def _cosyvoice_synthesize(text: str, voice_id: str, output_path: Path) -> None:
    try:
        from dashscope import Client
    except ImportError as exc:
        raise TTSError("未安装 dashscope SDK") from exc
    client = Client(api_key=DASHSCOPE_API_KEY, timeout=90.0, max_retries=0)
    synthesis = client.audio.tts.http.synthesize(
        text,
        model=TTS_MODEL,
        voice=voice_id,
        format="mp3",
        sample_rate=TTS_SAMPLE_RATE,
    )
    audio_url = str(getattr(synthesis, "audio_url", "") or "")
    if not audio_url:
        raise TTSError("CosyVoice 未返回 audio_url")
    with httpx.stream("GET", audio_url, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)
    if output_path.stat().st_size < 1000:
        raise TTSError("CosyVoice 返回音频为空")


def _voice_id_for_reference(reference_audio: Path | None) -> str:
    if reference_audio is None or not reference_audio.is_file():
        raise TTSError("缺少克隆参考音频")
    digest = hashlib.sha256(reference_audio.read_bytes()).hexdigest()[:20]
    cache_path = VOICE_CACHE_DIR / f"{digest}-{TTS_MODEL}.json"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("voice_id"):
            return str(payload["voice_id"])

    try:
        from dashscope import Client
    except ImportError as exc:
        raise TTSError("未安装 dashscope SDK") from exc
    reference_wav = VOICE_CACHE_DIR / f"{digest}-reference.wav"
    if not reference_wav.is_file():
        reference_wav.parent.mkdir(parents=True, exist_ok=True)
        _convert_to_wav(reference_audio, reference_wav)
    data_uri = f"data:audio/wav;base64,{base64.b64encode(reference_wav.read_bytes()).decode()}"
    client = Client(api_key=DASHSCOPE_API_KEY, timeout=90.0, max_retries=0)
    voice = client.audio.voices.create(
        target_model=TTS_MODEL,
        prefix="vpc",
        audio=data_uri,
    )
    voice_id = str(voice.voice_id)
    _wait_voice_ready(client, voice_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "voice_id": voice_id,
                "model": TTS_MODEL,
                "reference": str(reference_audio),
                "created_at": int(time.time()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return voice_id


def _wait_voice_ready(client: object, voice_id: str, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        voice = client.audio.voices.get(voice_id)
        status = str(getattr(voice, "status", "") or "")
        if status == "OK":
            return
        if status and status != "DEPLOYING":
            raise TTSError(f"克隆音色不可用：{status}")
        time.sleep(2)
    raise TTSError("克隆音色部署超时")


def _convert_to_wav(source: Path, target: Path) -> None:
    if FFMPEG_PATH is None:
        raise TTSError("未找到 FFmpeg")
    completed = subprocess.run(
        [
            str(FFMPEG_PATH),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0 or not target.is_file():
        raise TTSError(f"参考音频转换失败：{completed.stderr[-300:]}")


def _edge_synthesize(text: str, output_path: Path) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise TTSError("未安装 edge-tts") from exc

    async def run() -> None:
        await edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE).save(str(output_path))

    try:
        asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        raise TTSError(f"Edge TTS 失败：{exc}") from exc
    if not output_path.is_file() or output_path.stat().st_size < 1000:
        raise TTSError("Edge TTS 输出为空")
