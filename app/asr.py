from __future__ import annotations

import base64
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import (
    ASR_TIMEOUT_SECONDS,
    DASHSCOPE_API_KEY,
    DASHSCOPE_ASR_ENDPOINT,
    DASHSCOPE_ASR_MODEL,
    FFMPEG_PATH,
    FFPROBE_PATH,
)
from .srt import Segment

CLAUSE_END = "，。！？；：,.!?;:"


class AsrError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class AsrWord:
    text: str
    start_ms: int
    end_ms: int
    punctuation: str = ""


@dataclass
class AsrResult:
    text: str
    segments: list[Segment]
    words: list[AsrWord]
    duration_ms: int
    request_id: str | None
    model: str


def probe_duration_ms(video_path: Path) -> int:
    if FFPROBE_PATH is not None and video_path.is_file():
        completed = subprocess.run(
            [
                str(FFPROBE_PATH),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        try:
            return max(0, int(float(completed.stdout.strip()) * 1000))
        except ValueError:
            pass

    if FFMPEG_PATH is not None and video_path.is_file():
        completed = subprocess.run(
            [str(FFMPEG_PATH), "-hide_banner", "-i", str(video_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr)
        if match:
            hours, minutes, seconds = match.groups()
            return int(((int(hours) * 60 + int(minutes)) * 60 + float(seconds)) * 1000)
    return 0


def extract_audio(video_path: Path, work_dir: Path) -> tuple[Path, str]:
    """Extract a small mono track. MP3 keeps a 3-minute base64 payload around 1.4 MB."""
    if FFMPEG_PATH is None:
        raise AsrError("未找到 FFmpeg，请配置 FFMPEG_PATH")
    if not video_path.is_file():
        raise AsrError("视频文件不存在")

    attempts = [
        ("mp3", work_dir / "audio.mp3", ["-c:a", "libmp3lame", "-b:a", "48k"]),
        ("wav", work_dir / "audio.wav", ["-c:a", "pcm_s16le"]),
    ]
    errors: list[str] = []
    for audio_format, output_path, codec_args in attempts:
        completed = subprocess.run(
            [
                str(FFMPEG_PATH),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                *codec_args,
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 1024:
            return output_path, audio_format
        errors.append(completed.stderr.strip()[-400:])
        output_path.unlink(missing_ok=True)

    detail = errors[-1] if errors else "未知错误"
    if "does not contain any stream" in detail or "Output file #0 does not contain" in detail:
        raise AsrError("视频没有音轨，无法识别口播")
    raise AsrError(f"音频抽取失败：{detail}")


def transcribe_video(
    video_path: Path,
    work_dir: Path,
    *,
    model: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
) -> AsrResult:
    key = (api_key if api_key is not None else DASHSCOPE_API_KEY).strip()
    if not key:
        raise AsrError("未配置 DASHSCOPE_API_KEY")

    work_dir.mkdir(parents=True, exist_ok=True)
    duration_ms = probe_duration_ms(video_path)
    audio_path, audio_format = extract_audio(video_path, work_dir)
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    mime = "audio/mpeg" if audio_format == "mp3" else "audio/wav"
    data_url = f"data:{mime};base64,{encoded}"
    active_model = (model or DASHSCOPE_ASR_MODEL).strip()

    payload = {
        "model": active_model,
        "input": {
            "format": audio_format,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data_url, "format": audio_format},
                        }
                    ],
                }
            ],
        },
        "parameters": {
            "result_format": "message",
            "format": audio_format,
            "asr_options": {"enable_lid": True, "enable_itn": True},
        },
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = endpoint or DASHSCOPE_ASR_ENDPOINT

    response: httpx.Response | None = None
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(ASR_TIMEOUT_SECONDS, connect=20.0),
            )
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(1.5)
                continue
            if isinstance(exc, httpx.TimeoutException):
                raise AsrError("ASR 请求超时，请稍后重试", retryable=True) from exc
            raise AsrError("无法连接 ASR 服务，请检查网络后重试", retryable=True) from exc

        if response.status_code == 200:
            break
        if response.status_code >= 500 and attempt == 0:
            time.sleep(1.5)
            continue
        break

    if response is None:
        raise AsrError(f"ASR 请求失败：{last_error}", retryable=True)
    if response.status_code != 200:
        raise _http_error(response)

    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        raise AsrError("ASR 返回内容不是合法 JSON", retryable=True) from exc
    return parse_response(body, duration_ms=duration_ms, model=active_model)


def _http_error(response: httpx.Response) -> AsrError:
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = {}
    code = str(body.get("code") or "")
    message = str(body.get("message") or body.get("error", {}).get("message") or response.text[:300])

    if response.status_code in {401, 403}:
        return AsrError("DashScope 鉴权失败，请检查 DASHSCOPE_API_KEY", code=code)
    if code == "ASR_RESPONSE_HAVE_NO_WORDS":
        return AsrError("未识别到人声，请确认视频有口播，或改为上传 SRT/口播稿", code=code)
    if response.status_code == 429:
        return AsrError("ASR 请求被限流，请稍后重试", code=code, retryable=True)
    if response.status_code >= 500 or code == "SERVER_ERROR":
        return AsrError(f"ASR 服务临时异常：{message[:200]}", code=code, retryable=True)
    return AsrError(f"ASR 调用失败：{message[:200]}", code=code)


def parse_response(body: dict[str, Any], *, duration_ms: int, model: str) -> AsrResult:
    output = body.get("output") or {}
    inner = output.get("output") if isinstance(output.get("output"), dict) else output
    text = str(inner.get("text") or output.get("text") or "").strip()
    request_id = body.get("request_id") or output.get("request_id") or inner.get("request_id")

    sentence_value = inner.get("sentence") or output.get("sentence")
    if isinstance(sentence_value, dict):
        sentences = [sentence_value]
    elif isinstance(sentence_value, list):
        sentences = [item for item in sentence_value if isinstance(item, dict)]
    else:
        sentences = []

    effective_duration = duration_ms
    if not effective_duration:
        usage = body.get("usage") or output.get("usage") or {}
        try:
            effective_duration = int(float(usage.get("duration") or 0) * 1000)
        except (TypeError, ValueError):
            effective_duration = 0

    words: list[AsrWord] = []
    segments: list[Segment] = []
    for sentence in sentences:
        sentence_words: list[AsrWord] = []
        for item in sentence.get("words") or []:
            if not isinstance(item, dict):
                continue
            word = AsrWord(
                text=str(item.get("text") or ""),
                start_ms=int(item.get("begin_time") or 0),
                end_ms=int(item.get("end_time") or 0),
                punctuation=str(item.get("punctuation") or ""),
            )
            sentence_words.append(word)
            words.append(word)

        sentence_text = str(sentence.get("text") or "").strip()
        if sentence_words:
            sentence_segments = segments_from_words(
                sentence_words,
                effective_duration,
                display_text=sentence_text,
            )
        elif sentence_text:
            start_ms = int(sentence.get("begin_time") or 0)
            end_ms = int(sentence.get("end_time") or effective_duration or 0)
            sentence_segments = [
                Segment(
                    index=1,
                    start_ms=start_ms,
                    end_ms=max(end_ms, start_ms + 300),
                    text=sentence_text,
                )
            ]
        else:
            sentence_segments = []

        for segment in sentence_segments:
            segments.append(
                Segment(
                    index=len(segments) + 1,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=segment.text,
                )
            )

    if not segments and text:
        segments = segments_from_text(text, effective_duration)
    if not segments:
        raise AsrError(
            "未识别到人声，请确认视频有口播，或改为上传 SRT/口播稿",
            code="ASR_RESPONSE_HAVE_NO_WORDS",
        )

    if not text:
        text = "\n".join(segment.text for segment in segments)
    if effective_duration <= 0 and segments:
        effective_duration = segments[-1].end_ms
    return AsrResult(
        text=text,
        segments=segments,
        words=words,
        duration_ms=effective_duration,
        request_id=str(request_id) if request_id else None,
        model=model,
    )


@dataclass
class _TimedToken:
    text: str
    start_ms: int
    end_ms: int
    display_start: int | None = None
    display_end: int | None = None


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(text):
        if re.match(r"[\s，。！？；：、,.!?;:·\"'“”‘’（）()\[\]【】]", char):
            continue
        chars.append(char.lower())
        mapping.append(index)
    return "".join(chars), mapping


def segments_from_words(
    words: list[AsrWord],
    duration_ms: int,
    display_text: str | None = None,
) -> list[Segment]:
    display = display_text or ""
    normalized_display, display_map = _normalize_with_map(display)
    cursor = 0
    tokens: list[_TimedToken] = []
    for word in words:
        token_text = f"{word.text}{word.punctuation}"
        normalized_word, _ = _normalize_with_map(token_text)
        display_start: int | None = None
        display_end: int | None = None
        if normalized_word:
            position = normalized_display.find(normalized_word, cursor)
            if position >= 0:
                display_start = display_map[position]
                display_end = display_map[position + len(normalized_word) - 1] + 1
                cursor = position + len(normalized_word)
        tokens.append(
            _TimedToken(
                text=token_text,
                start_ms=word.start_ms,
                end_ms=word.end_ms,
                display_start=display_start,
                display_end=display_end,
            )
        )

    if not tokens:
        return []

    segments: list[Segment] = []
    buffer: list[_TimedToken] = []
    buffer_start: int | None = None
    buffer_chars = 0

    def flush(end_ms: int, display_boundary: int | None) -> None:
        nonlocal buffer, buffer_start, buffer_chars
        if (
            display
            and buffer
            and buffer[0].display_start is not None
            and display_boundary is not None
            and all(token.display_start is not None for token in buffer)
        ):
            line = display[buffer[0].display_start : display_boundary].strip()
        else:
            line = "".join(token.text for token in buffer).strip()
        if line and buffer_start is not None:
            end = max(end_ms, buffer_start + 200)
            if duration_ms:
                end = min(end, duration_ms)
            segments.append(
                Segment(index=len(segments) + 1, start_ms=buffer_start, end_ms=end, text=line)
            )
        buffer = []
        buffer_start = None
        buffer_chars = 0

    for index, token in enumerate(tokens):
        if not token.text.strip():
            continue
        if buffer_start is None:
            buffer_start = token.start_ms
        buffer.append(token)
        buffer_chars += len(token.text.strip())

        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        next_start = next_token.start_ms if next_token is not None else None
        gap = next_start - token.end_ms if next_start is not None else 0
        between = ""
        if (
            display
            and token.display_end is not None
            and next_token is not None
            and next_token.display_start is not None
        ):
            between = display[token.display_end : next_token.display_start]
        clause_end = token.text.rstrip().endswith(tuple(CLAUSE_END)) or any(
            char in CLAUSE_END for char in between
        )
        too_long = buffer_chars >= 32
        if clause_end or gap >= 600 or too_long:
            if display and next_token is None:
                boundary = len(display)
            elif display and next_token is not None and next_token.display_start is not None:
                boundary = next_token.display_start
            elif display and token.display_end is not None:
                boundary = token.display_end
            else:
                boundary = None
            flush(token.end_ms, boundary)

    if buffer:
        boundary = len(display) if display else None
        flush(tokens[-1].end_ms, boundary)
    return segments


def segments_from_text(text: str, duration_ms: int) -> list[Segment]:
    pieces = [piece.strip() for piece in re.split(r"(?<=[。！？!?；;])", text) if piece.strip()]
    if not pieces:
        pieces = [text.strip()]

    lines: list[str] = []
    for piece in pieces:
        if len(piece) <= 28:
            lines.append(piece)
            continue
        for index in range(0, len(piece), 24):
            chunk = piece[index : index + 24].strip()
            if chunk:
                lines.append(chunk)

    total_chars = sum(len(line) for line in lines) or 1
    effective_duration = duration_ms or total_chars * 260
    segments: list[Segment] = []
    cursor = 0
    for index, line in enumerate(lines):
        if index == len(lines) - 1:
            end = effective_duration
        else:
            end = cursor + max(400, int(effective_duration * len(line) / total_chars))
        segments.append(
            Segment(
                index=index + 1,
                start_ms=cursor,
                end_ms=max(end, cursor + 200),
                text=line,
            )
        )
        cursor = end
    return segments
