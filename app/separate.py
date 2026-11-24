"""人声 / BGM 分离与磁盘缓存。

Demucs 依赖 torch，体积大，不放进主依赖；这里通过外部解释器调用，路径由
`DEMUCS_PYTHON` 指定。同一文件的同一模型只会分离一次，结果按内容 hash 缓存。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import (
    DATA_DIR,
    DEMUCS_DEVICE,
    DEMUCS_MODEL,
    DEMUCS_PYTHON,
    DEMUCS_TIMEOUT_SECONDS,
    FFMPEG_PATH,
    STEMS_DIR,
)

Runner = Callable[[Path, Path], tuple[Path, Path]]


class SeparateError(RuntimeError):
    """分离链路不可用或执行失败。"""


@dataclass(frozen=True)
class Stems:
    vocals: Path
    background: Path
    cache_key: str
    cached: bool
    source: Path


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(source: Path, *, model: str, device: str) -> str:
    return f"{file_digest(source)[:20]}-{model}-{device}".replace("/", "_")


def extract_wav(source: Path, dest: Path) -> None:
    if FFMPEG_PATH is None:
        raise SeparateError("未找到 ffmpeg，无法抽取音频")
    completed = subprocess.run(
        [
            str(FFMPEG_PATH), "-y", "-v", "error", "-i", str(source),
            "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dest),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0 or not dest.is_file() or dest.stat().st_size < 1024:
        raise SeparateError(f"音频抽取失败：{completed.stderr[-300:]}")


def run_demucs(
    wav: Path,
    work_dir: Path,
    *,
    python: Path | None = None,
    model: str | None = None,
    device: str | None = None,
    timeout: float | None = None,
) -> tuple[Path, Path]:
    interpreter = python or DEMUCS_PYTHON
    if interpreter is None or not Path(interpreter).is_file():
        raise SeparateError("未配置可用的 DEMUCS_PYTHON，无法调用 Demucs")
    out_dir = work_dir / "demucs"
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(DATA_DIR / "tools" / "torch-home"))
    # 没有外网时 huggingface_hub 会重试 5 次，动辄卡两分钟；模型定义本地已有。
    env["HF_HUB_OFFLINE"] = "1"
    completed = subprocess.run(
        [
            str(interpreter), "-m", "demucs",
            "-d", device or DEMUCS_DEVICE,
            "--two-stems=vocals",
            "-n", model or DEMUCS_MODEL,
            "-o", str(out_dir),
            str(wav),
        ],
        capture_output=True,
        text=True,
        timeout=timeout or DEMUCS_TIMEOUT_SECONDS,
        env=env,
        check=False,
    )
    vocals = next(out_dir.rglob("vocals.wav"), None)
    background = next(out_dir.rglob("no_vocals.wav"), None)
    if completed.returncode != 0 or vocals is None or background is None:
        raise SeparateError(f"Demucs 分离失败：{completed.stderr[-400:]}")
    return vocals, background


def separate_audio(
    source: Path,
    *,
    model: str | None = None,
    device: str | None = None,
    use_cache: bool = True,
    runner: Runner | None = None,
    stems_dir: Path | None = None,
) -> Stems:
    """把视频的人声和 BGM 分成两条 wav，按文件内容缓存。"""
    source = Path(source)
    if not source.is_file():
        raise SeparateError(f"源文件不存在：{source}")

    active_model = model or DEMUCS_MODEL
    active_device = device or DEMUCS_DEVICE
    key = cache_key(source, model=active_model, device=active_device)
    root = Path(stems_dir or STEMS_DIR) / key
    vocals_path = root / "vocals.wav"
    background_path = root / "no_vocals.wav"

    if use_cache and vocals_path.is_file() and background_path.is_file():
        return Stems(vocals_path, background_path, key, True, source)

    root.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = root.parent / f".tmp-{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True)
    try:
        wav = temp_dir / "input.wav"
        extract_wav(source, wav)
        active_runner = runner or (lambda path, work: run_demucs(path, work, model=active_model, device=active_device))
        vocals, background = active_runner(wav, temp_dir)
        for produced, target in (
            (Path(vocals), temp_dir / "vocals.wav"),
            (Path(background), temp_dir / "no_vocals.wav"),
        ):
            if produced.resolve() != target.resolve():
                shutil.copy2(produced, target)
        wav.unlink(missing_ok=True)
        (temp_dir / "meta.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "size": source.stat().st_size,
                    "model": active_model,
                    "device": active_device,
                    "created_at": int(time.time()),
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        if root.exists():
            shutil.rmtree(root)
        temp_dir.replace(root)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return Stems(root / "vocals.wav", root / "no_vocals.wav", key, False, source)
