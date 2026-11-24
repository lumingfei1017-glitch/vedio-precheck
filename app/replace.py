"""整镜替换服务层：切参考 → 上传 → 生成 → 下载 → 回贴。

CLI 与后台任务共用这一层，避免两套 ffmpeg / 接口调用逻辑。
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .asr import probe_duration_ms
from .config import (
    FFMPEG_PATH,
    FFPROBE_PATH,
    SEGMENT_PRESERVE_BGM,
    VOICE_CACHE_DIR,
    VOICE_REFERENCE_TARGET_MS,
)
from .edit import (
    extract_stable_voice_reference,
    extract_voice_reference,
    replace_audio_interval,
)
from .separate import file_digest, separate_audio
from .tts import TTSError, fit_audio, synthesize
from .wan import (
    CHARS_PER_SECOND,
    MAX_WAN_CHARS_PER_SECOND,
    WAN_DURATION_ROUND_UP_MAX_RATIO,
    build_shot_prompt,
    char_budget,
    download_result,
    planned_duration,
    poll_task,
    submit_task,
    text_fit,
    upload_media,
)

Progress = Callable[[str, int], None]


class ReplaceError(RuntimeError):
    """整镜替换失败。"""


@dataclass(frozen=True)
class ReplacePlan:
    start: float
    end: float
    text: str
    chars: int
    budget: int
    duration: float
    fit: str
    route: str = "wan"

    @property
    def warnings(self) -> list[str]:
        notes: list[str] = []
        if self.chars > self.budget:
            notes.append(f"文案超出预算 {self.chars - self.budget} 字，语速会很赶")
        if self.fit == "short":
            notes.append("文案偏短，镜头末尾会留空档，容易被原句碎片填空")
        elif self.fit == "long":
            notes.append("文案偏长，建议精简")
        return notes

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "ref_seconds": round(self.end - self.start, 2),
            "chars": self.chars,
            "char_budget": self.budget,
            "duration": self.duration,
            "fit": self.fit,
            "route": self.route,
            "warnings": self.warnings,
        }


def plan_replace(start: float, end: float, text: str) -> ReplacePlan:
    if end <= start:
        raise ReplaceError("镜头区间无效")
    if end - start < 0.5:
        raise ReplaceError("镜头区间太短")
    clean = text.strip()
    if not clean:
        raise ReplaceError("缺少新台词")
    duration = planned_duration(end - start)
    if duration > (end - start) * (1 + WAN_DURATION_ROUND_UP_MAX_RATIO) + 0.05:
        raise ReplaceError("镜头太短，Wan 输出时长会超过参考片段，请改走 TTS/CUT 或人工处理")
    # 文案按 Wan 实际生成时长（整数秒）来配，参考片段常带小数，按参考配会偏密。
    budget = char_budget(duration)
    hard_capacity = int(duration * MAX_WAN_CHARS_PER_SECOND)
    if len(clean) > hard_capacity:
        raise ReplaceError(
            f"文案 {len(clean)} 字约 {len(clean) / CHARS_PER_SECOND:.1f}s，"
            f"超过本镜 {duration}s 的容量（建议 {budget} 字，上限 {hard_capacity} 字）；"
            "直接生成会吞字糊音，请先精简"
        )
    return ReplacePlan(
        start=start,
        end=end,
        text=clean,
        chars=len(clean),
        budget=budget,
        duration=duration,
        fit=text_fit(len(clean), duration),
        route="wan",
    )


def plan_local_tts(start: float, end: float, text: str) -> ReplacePlan:
    """Validate a local TTS edit that keeps the original shot duration."""
    if end <= start:
        raise ReplaceError("镜头区间无效")
    if end - start < 0.5:
        raise ReplaceError("镜头区间太短")
    clean = text.strip()
    if not clean:
        raise ReplaceError("缺少新台词")
    seconds = end - start
    return ReplacePlan(
        start=start,
        end=end,
        text=clean,
        chars=len(clean),
        budget=char_budget(seconds),
        duration=round(seconds, 3),
        fit=text_fit(len(clean), seconds),
        route="tts",
    )


def validate_generated_duration(
    generated_ms: int,
    reference_ms: int,
    *,
    tolerance_ms: int = 80,
    max_ratio: float = 0.08,
) -> None:
    limit = reference_ms * (1 + max_ratio) + tolerance_ms
    if generated_ms > limit:
        raise ReplaceError(
            f"Wan 输出 {generated_ms / 1000:.2f}s，超过参考镜头 "
            f"{reference_ms / 1000:.2f}s，超过 {max_ratio:.0%} 的回贴变速上限，"
            "已拒绝回贴以避免发糊吞字"
        )


def _run_ffmpeg(args: list[str], label: str) -> None:
    if FFMPEG_PATH is None:
        raise ReplaceError("未找到 ffmpeg")
    completed = subprocess.run(
        [str(FFMPEG_PATH), "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ReplaceError(f"{label}失败：{completed.stderr[-400:]}")


def _probe(video: Path) -> dict:
    if FFPROBE_PATH is None:
        raise ReplaceError("未找到 ffprobe")
    out = subprocess.run(
        [
            str(FFPROBE_PATH), "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
            "-of", "json", str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def cut_reference(video: Path, start: float, end: float, work: Path) -> Path:
    ref = work / "reference.mp4"
    _run_ffmpeg(
        [
            "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
            "-vf",
            "scale='if(gt(iw,4096),4096,iw)':'if(gt(ih,4096),4096,ih)':"
            "force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(ref),
        ],
        "参考片段裁剪",
    )
    return ref


def cut_reference_audio(video: Path, start: float, end: float, work: Path) -> Path:
    audio = work / "reference.wav"
    _run_ffmpeg(
        [
            "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio),
        ],
        "参考音频抽取",
    )
    return audio


def build_splice_graph(
    *,
    width: int,
    height: int,
    fps: str,
    start: float,
    end: float,
    generated_ms: int,
    background: bool = False,
) -> tuple[str, float, float | None]:
    """生成回贴滤镜，把生成片段补齐或裁到与原片槽位完全等长。

    Wan 的 duration 只收整数秒，槽位几乎都带小数，直接拼接会让时间轴变长或变短。
    差额在 ±8% 以内时用变速填满（不丢字、不冻结），超出才补帧或裁剪。
    """
    slot = max(0.0, end - start)
    generated = max(0.001, generated_ms / 1000)
    ratio = generated / slot if slot > 0 else 1.0
    stretch: float | None = None
    if 0.92 <= ratio <= 1.08 and abs(ratio - 1.0) > 0.002:
        stretch = ratio
        pad = 0.0
        video_chain = (
            f"setpts=(PTS-STARTPTS)*{slot / generated:.6f},"
            "tpad=stop_mode=clone:stop_duration=0.060,"
        )
        audio_chain = f"asetpts=PTS-STARTPTS,atempo={ratio:.6f},apad=pad_dur=0.060,"
    else:
        pad = max(0.0, slot - generated)
        video_chain = "setpts=PTS-STARTPTS,"
        audio_chain = "asetpts=PTS-STARTPTS,"
    video_pad = f"tpad=stop_mode=clone:stop_duration={pad:.3f}," if pad > 0.001 else ""
    audio_pad = f"apad=pad_dur={pad:.3f}," if pad > 0.001 else ""
    graph = (
        f"[0:v]trim=0:{start:.3f},setpts=PTS-STARTPTS[v0];"
        f"[0:a]atrim=0:{start:.3f},aresample=48000,asetpts=PTS-STARTPTS[a0];"
        f"[1:v]scale={width}:{height},fps={fps},setsar=1,{video_chain}"
        f"{video_pad}trim=0:{slot:.3f},setpts=PTS-STARTPTS[v1];"
        f"[1:a]aresample=48000,{audio_chain}"
        f"{audio_pad}atrim=0:{slot:.3f},asetpts=PTS-STARTPTS[voice];"
        + (
            f"[2:a]atrim={start:.3f}:{end:.3f},aresample=48000,asetpts=PTS-STARTPTS[bg];"
            "[bg][voice]amix=inputs=2:duration=first:normalize=0[a1];"
            if background
            else "[voice]anull[a1];"
        )
        +
        f"[0:v]trim={end:.3f},setpts=PTS-STARTPTS[v2];"
        f"[0:a]atrim={end:.3f},aresample=48000,asetpts=PTS-STARTPTS[a2];"
        f"[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]"
    )
    return graph, pad, stretch


def splice(
    video: Path,
    generated: Path,
    start: float,
    end: float,
    out: Path,
    *,
    background: Path | None = None,
) -> tuple[float, float | None]:
    info = _probe(video)
    streams = {item["codec_type"]: item for item in info["streams"]}
    width, height = streams["video"]["width"], streams["video"]["height"]
    fps = streams["video"]["r_frame_rate"]
    width -= width % 2
    height -= height % 2
    graph, pad, stretch = build_splice_graph(
        width=width,
        height=height,
        fps=fps,
        start=start,
        end=end,
        generated_ms=probe_duration_ms(generated),
        background=background is not None,
    )
    command = ["-i", str(video), "-i", str(generated)]
    if background is not None:
        command.extend(["-i", str(background)])
    command.extend(
        [
            "-filter_complex", graph,
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-pix_fmt", "yuv420p", "-r", fps, "-c:a", "aac",
            "-b:a", "160k", "-movflags", "+faststart", str(out),
        ]
    )
    _run_ffmpeg(command, "回贴合成")
    return pad, stretch


def replace_shot(
    video: Path,
    start: float,
    end: float,
    text: str,
    out: Path,
    *,
    work_dir: Path | None = None,
    scene: str | None = None,
    action: str | None = None,
    prompt: str | None = None,
    resolution: str = "720P",
    progress: Progress | None = None,
) -> dict:
    original_start, original_end = start, end
    try:
        from .boundary import snap_window

        snapped_start, snapped_end = snap_window(
            video,
            int(round(start * 1000)),
            int(round(end * 1000)),
        )
        if snapped_end > snapped_start + 500:
            start, end = snapped_start / 1000, snapped_end / 1000
    except Exception:  # noqa: BLE001
        pass
    report = plan_replace(start, end, text)
    work = Path(work_dir) if work_dir else Path(out).parent / "work"
    work.mkdir(parents=True, exist_ok=True)

    def emit(stage: str, value: int) -> None:
        if progress:
            progress(stage, value)

    emit("切参考片段", 8)
    ref = cut_reference(video, start, end, work)
    audio = cut_reference_audio(video, start, end, work)
    emit("上传素材", 15)
    video_url = upload_media(ref)
    audio_url = upload_media(audio)
    active_prompt = prompt or build_shot_prompt(
        report.text,
        scene=scene or "室内竖屏服装展示口播",
        action=action or "她面向镜头站立介绍身上的产品，手部动作自然，镜头景别保持不变",
    )
    emit("提交生成任务", 22)
    task_id = submit_task(
        [{"type": "reference_video", "url": video_url}, {"type": "reference_audio", "url": audio_url}],
        active_prompt,
        report.duration,
        resolution=resolution,
    )
    emit("等待 Wan 生成", 30)
    started = time.time()
    body = poll_task(task_id)
    generated = download_result(body["output"]["video_url"], work / "generated.mp4")
    generated_ms = probe_duration_ms(generated)
    reference_ms = int(round((end - start) * 1000))
    validate_generated_duration(generated_ms, reference_ms)
    background: Path | None = None
    bgm_warning: str | None = None
    if SEGMENT_PRESERVE_BGM:
        emit("分离原片伴奏", 80)
        try:
            background = separate_audio(video).background
        except Exception as exc:  # noqa: BLE001
            bgm_warning = f"BGM 未保留：人声分离不可用（{str(exc)[:160]}）"
    emit("回贴合成", 85)
    padded, stretch = splice(
        video, generated, start, end, Path(out), background=background
    )
    emit("完成", 100)
    warnings = list(report.warnings)
    if bgm_warning:
        warnings.append(bgm_warning)
    if stretch is not None:
        warnings.append(
            f"生成片段 {generated_ms / 1000:.2f}s 与槽位 {reference_ms / 1000:.2f}s 不等长，"
            f"已按 {stretch:.3f} 倍轻微变速对齐（未裁剪、未冻结）"
        )
    elif padded > 0.05:
        warnings.append(
            f"Wan 输出 {generated_ms / 1000:.2f}s 短于参考 {reference_ms / 1000:.2f}s，"
            f"已在末尾冻结补帧 {padded * 1000:.0f}ms 以保持总时长"
        )
    return {
        **report.as_dict(),
        "start": round(start, 3),
        "end": round(end, 3),
        "snapped_from": (
            {"start": round(original_start, 3), "end": round(original_end, 3)}
            if (round(start, 3), round(end, 3)) != (round(original_start, 3), round(original_end, 3))
            else None
        ),
        "warnings": warnings,
        "task_id": task_id,
        "generated": str(generated),
        "generated_duration_ms": generated_ms,
        "reference_duration_ms": reference_ms,
        "padded_ms": int(round(padded * 1000)),
        "stretch_ratio": round(stretch, 4) if stretch is not None else None,
        "bgm_preserved": background is not None,
        "output": str(out),
        "elapsed_seconds": round(time.time() - started, 1),
        "usage": body.get("usage"),
    }


def replace_shot_tts(
    video: Path,
    start: float,
    end: float,
    text: str,
    out: Path,
    *,
    work_dir: Path | None = None,
    progress: Progress | None = None,
) -> dict:
    """Replace only a shot's voice track, keeping its original BGM."""
    report = plan_local_tts(start, end, text)
    work = Path(work_dir) if work_dir else Path(out).parent / "work"
    work.mkdir(parents=True, exist_ok=True)

    def emit(stage: str, value: int) -> None:
        if progress:
            progress(stage, value)

    emit("分离原人声与 BGM", 8)
    try:
        stems = separate_audio(video)
    except Exception as exc:  # noqa: BLE001
        raise ReplaceError(f"无法分离人声与 BGM，局部换音轨不可用：{exc}") from exc

    duration_ms = probe_duration_ms(video)
    start_ms = int(round(start * 1000))
    end_ms = int(round(end * 1000))
    reference: Path | None = None
    warnings = list(report.warnings)
    try:
        stable_reference = (
            VOICE_CACHE_DIR
            / f"{file_digest(video)[:16]}-voice-reference-{VOICE_REFERENCE_TARGET_MS}.wav"
        )
        if not stable_reference.is_file() or stable_reference.stat().st_size < 100_000:
            stable_reference = extract_stable_voice_reference(
                stems.vocals,
                stable_reference,
            )
        reference = stable_reference
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"稳定参考音不可用，改用临时参考：{str(exc)[:180]}")
        try:
            reference = extract_voice_reference(
                video,
                start_ms,
                end_ms,
                duration_ms,
                work / "voice-reference.wav",
                source_audio=stems.vocals,
            )
        except Exception:
            reference = None

    emit("生成局部配音", 34)
    raw_audio = work / "shot-tts.mp3"
    fitted_audio = work / "shot-fit.mp3"
    try:
        tts_result = synthesize(report.text, raw_audio, reference_audio=reference)
        _, tempo, fit_warnings = fit_audio(
            raw_audio,
            fitted_audio,
            max(1, end_ms - start_ms),
        )
        warnings.extend(fit_warnings)
    except TTSError as exc:
        raise ReplaceError(f"局部配音失败：{exc}") from exc

    if tts_result.voice_source != "cloned":
        warnings.append("未使用原人物音色，当前为 Edge 测试音色")

    emit("回贴原 BGM 与视频", 78)
    replace_audio_interval(
        video,
        fitted_audio,
        start_ms,
        end_ms,
        duration_ms,
        Path(out),
        background_audio=stems.background,
    )
    emit("完成", 100)
    return {
        **report.as_dict(),
        "mode": "tts",
        "output": str(out),
        "duration_ms": duration_ms,
        "provider": tts_result.provider,
        "voice_id": tts_result.voice_id,
        "voice_source": tts_result.voice_source,
        "voice_reference": str(reference) if reference else None,
        "tempo": round(tempo, 3),
        "bgm_preserved": True,
        "warnings": list(dict.fromkeys([*warnings, *tts_result.warnings])),
    }
