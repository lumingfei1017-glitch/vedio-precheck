"""把人声边界吸附到真实静音处。

ASR 给的句子起止是"识别边界"，不是"发声边界"：实测第一条素材 ASR 说 0.56s 开始，
实际 0.02s 就出声了；第二条 39.83s 的切点正好落在"腹部"两个字中间。
直接按 ASR 边界替换会漏出半句原声，或者把词切一半。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .config import FFMPEG_PATH

SAMPLE_RATE = 16000
FRAME_MS = 20
# 相对阈值：比搜索窗内第 10 百分位的能量高出 8dB 以下算静音。
SILENCE_ABOVE_P10_DB = 8.0
MIN_SILENCE_MS = 100
SEARCH_MS = 600
LEAD_MARGIN_MS = 60
TAIL_MARGIN_MS = 80

_PCM_CACHE: dict[tuple[str, int], np.ndarray] = {}


def load_pcm(video: Path) -> np.ndarray:
    try:
        stat = Path(video).stat()
    except OSError:
        return np.zeros(0)
    key = (str(video), int(stat.st_mtime_ns))
    cached = _PCM_CACHE.get(key)
    if cached is not None:
        return cached
    if FFMPEG_PATH is None:
        return np.zeros(0)
    raw = subprocess.run(
        [
            str(FFMPEG_PATH), "-v", "error", "-i", str(video),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
        ],
        capture_output=True,
        timeout=180,
        check=False,
    ).stdout
    samples = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float64) / 32768
    _PCM_CACHE[key] = samples
    return samples


def frame_db(samples: np.ndarray) -> np.ndarray:
    hop = int(SAMPLE_RATE * FRAME_MS / 1000)
    if len(samples) < hop:
        return np.zeros(0)
    count = len(samples) // hop
    trimmed = samples[: count * hop].reshape(count, hop)
    rms = np.sqrt(np.mean(trimmed**2, axis=1))
    with np.errstate(divide="ignore"):
        return 20 * np.log10(np.maximum(rms, 1e-9))


def _silence_threshold(db: np.ndarray) -> float:
    if not len(db):
        return -40.0
    return float(np.percentile(db, 10)) + SILENCE_ABOVE_P10_DB


def _silent_runs(db: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """返回长度达到 MIN_SILENCE_MS 的静音区间（单位：帧）。"""
    need = max(1, MIN_SILENCE_MS // FRAME_MS)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(db):
        if value <= threshold:
            if start is None:
                start = index
        else:
            if start is not None and index - start >= need:
                runs.append((start, index))
            start = None
    if start is not None and len(db) - start >= need:
        runs.append((start, len(db)))
    return runs


def snap_start(video: Path, time_ms: int, *, search_ms: int = SEARCH_MS) -> int:
    """向前走到真正的静音处，保证不把上一句的起音留下来。"""
    db = frame_db(load_pcm(video))
    if not len(db):
        return max(0, time_ms)
    center = min(len(db) - 1, max(0, int(time_ms / FRAME_MS)))
    low = max(0, int((time_ms - search_ms) / FRAME_MS))
    high = min(len(db), center + 10)
    window = db[low:high]
    threshold = _silence_threshold(window)
    runs = _silent_runs(window, threshold)
    inside = [run for run in runs if run[0] <= center - low < run[1]]
    if inside:
        # 切点本来就落在静音里，保持不动。
        return time_ms
    before = [run for run in runs if run[1] <= center - low + 1]
    if before:
        end = before[-1][1] + low
        return max(0, end * FRAME_MS - LEAD_MARGIN_MS)
    # 搜索窗内一直有人声：这句从更早的地方就开始了，退到搜索窗起点。
    return low * FRAME_MS


def snap_end(video: Path, time_ms: int, *, search_ms: int = SEARCH_MS) -> int:
    """向后走到真正的静音处，保证不把尾音切掉。"""
    db = frame_db(load_pcm(video))
    if not len(db):
        return time_ms
    center = min(len(db) - 1, max(0, int(time_ms / FRAME_MS)))
    low = max(0, center - 10)
    high = min(len(db), int((time_ms + search_ms) / FRAME_MS))
    window = db[low:high]
    threshold = _silence_threshold(window)
    runs = _silent_runs(window, threshold)
    inside = [run for run in runs if run[0] <= center - low < run[1]]
    if inside:
        return time_ms
    after = [run for run in runs if run[0] >= center - low - 1]
    if after:
        start = after[0][0] + low
        return int(start * FRAME_MS + TAIL_MARGIN_MS)
    return time_ms


def snap_window(video: Path, start_ms: int, end_ms: int) -> tuple[int, int]:
    start = snap_start(video, start_ms)
    end = snap_end(video, end_ms)
    if end <= start + 500:
        return start_ms, end_ms
    return start, end
