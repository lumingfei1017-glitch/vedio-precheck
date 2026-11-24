from __future__ import annotations

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding the real environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.environ.get("VIDEO_PRECHECK_DATA_DIR") or BASE_DIR / "data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = Path(os.environ.get("VIDEO_PRECHECK_DB") or DATA_DIR / "app.db")
RULES_PATH = BASE_DIR / "rules" / "qianchuan_apparel.json"
GENERATED_RULES_PATH = BASE_DIR / "rules" / "generated_keywords.json"
PROMPT_HINTS_PATH = BASE_DIR / "rules" / "prompt_hints.json"
STATIC_DIR = BASE_DIR / "static"

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_ASR_BASE_URL = os.environ.get(
    "DASHSCOPE_ASR_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
).strip()
DASHSCOPE_ASR_MODEL = os.environ.get(
    "DASHSCOPE_ASR_MODEL", "qwen-audio-3.0-asr-flash"
).strip()
ASR_TIMEOUT_SECONDS = float(os.environ.get("ASR_TIMEOUT_SECONDS", "300"))

DASHSCOPE_BASE_URL = os.environ.get(
    "BASE_URL1", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).strip()
SEMANTIC_ENABLED = os.environ.get("SEMANTIC_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
SEMANTIC_BASE_URL = os.environ.get("SEMANTIC_BASE_URL", DASHSCOPE_BASE_URL).strip()
SEMANTIC_MODEL = os.environ.get("SEMANTIC_MODEL", "deepseek-v4.1-flash-1017").strip()
SEMANTIC_TIMEOUT_SECONDS = float(os.environ.get("SEMANTIC_TIMEOUT_SECONDS", "90"))
SEMANTIC_MAX_TOKENS = int(os.environ.get("SEMANTIC_MAX_TOKENS", "6000"))


def semantic_endpoint(base_url: str) -> str:
    root = base_url.strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return f"{root}/chat/completions"


SEMANTIC_ENDPOINT = semantic_endpoint(SEMANTIC_BASE_URL)

VISUAL_ENABLED = os.environ.get("VISUAL_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
VISUAL_MODEL = os.environ.get("VISUAL_MODEL", "qwen3.8-flash").strip()
VISUAL_TIMEOUT_SECONDS = float(os.environ.get("VISUAL_TIMEOUT_SECONDS", "120"))
VISUAL_MAX_TOKENS = int(os.environ.get("VISUAL_MAX_TOKENS", "1800"))
VISUAL_MAX_FRAMES = int(os.environ.get("VISUAL_MAX_FRAMES", "16"))
VISUAL_FRAME_WIDTH = int(os.environ.get("VISUAL_FRAME_WIDTH", "512"))
VISUAL_ENDPOINT = semantic_endpoint(DASHSCOPE_BASE_URL)
# 视觉事件二次定位：粗扫每 5–6 秒一帧，事件时间窗会跟着失真，命中后再密集复核。
VISUAL_RELOCALIZE_ENABLED = os.environ.get(
    "VISUAL_RELOCALIZE_ENABLED", "true"
).strip().lower() not in {"0", "false", "no"}
VISUAL_RELOCALIZE_RADIUS_MS = int(os.environ.get("VISUAL_RELOCALIZE_RADIUS_MS", "12000"))
VISUAL_RELOCALIZE_STEP_MS = int(os.environ.get("VISUAL_RELOCALIZE_STEP_MS", "1000"))
VISUAL_RELOCALIZE_MAX_FRAMES = int(os.environ.get("VISUAL_RELOCALIZE_MAX_FRAMES", "24"))
VISUAL_RELOCALIZE_MAX_EVENTS = int(os.environ.get("VISUAL_RELOCALIZE_MAX_EVENTS", "6"))
VISUAL_OWN_BRANDS = [
    item.strip()
    for item in os.environ.get("VISUAL_OWN_BRANDS", "北极绒").split(",")
    if item.strip()
]

REWRITE_ENABLED = os.environ.get("REWRITE_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "deepseek-v4.1-flash-1017").strip()
REWRITE_TIMEOUT_SECONDS = float(os.environ.get("REWRITE_TIMEOUT_SECONDS", "90"))
REWRITE_MAX_TOKENS = int(os.environ.get("REWRITE_MAX_TOKENS", "4000"))
REWRITE_ENDPOINT = semantic_endpoint(DASHSCOPE_BASE_URL)

RENDER_ENABLED = os.environ.get("RENDER_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
TTS_VOICE = os.environ.get("TTS_VOICE", "zh-CN-XiaoxiaoNeural").strip()
TTS_RATE = os.environ.get("TTS_RATE", "+0%").strip()
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "auto").strip().lower()
TTS_MODEL = os.environ.get("TTS_MODEL", "cosyvoice-v3-plus").strip()
TTS_VOICE_ID = os.environ.get("TTS_VOICE_ID", "").strip()
VOICE_CLONE_ENABLED = os.environ.get("VOICE_CLONE_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
VOICE_REFERENCE_PATH = os.environ.get("VOICE_REFERENCE_PATH", "").strip()
TTS_MAX_TEMPO = float(os.environ.get("TTS_MAX_TEMPO", "1.15"))
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
VOICE_CACHE_DIR = Path(os.environ.get("VOICE_CACHE_DIR") or DATA_DIR / "voices")
SEGMENT_REPLACE_ENABLED = os.environ.get("SEGMENT_REPLACE_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
SEGMENT_PRESERVE_BGM = os.environ.get("SEGMENT_PRESERVE_BGM", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
VOICE_REFERENCE_MIN_MS = int(os.environ.get("VOICE_REFERENCE_MIN_MS", "8000"))
VOICE_REFERENCE_MAX_MS = int(os.environ.get("VOICE_REFERENCE_MAX_MS", "30000"))
VOICE_REFERENCE_TARGET_MS = int(os.environ.get("VOICE_REFERENCE_TARGET_MS", "30000"))
SHOT_TTS_FALLBACK_ENABLED = os.environ.get(
    "SHOT_TTS_FALLBACK_ENABLED", "false"
).strip().lower() not in {
    "0",
    "false",
    "no",
}

# 只保留 Wan 纯口播改写：局部换音（TTS）与自动删段默认关闭，其余转人工。
TTS_ROUTE_ENABLED = os.environ.get("TTS_ROUTE_ENABLED", "false").strip().lower() not in {
    "0",
    "false",
    "no",
}
AUTO_CUT_ENABLED = os.environ.get("AUTO_CUT_ENABLED", "false").strip().lower() not in {
    "0",
    "false",
    "no",
}
BATCH_RENDER_ENABLED = os.environ.get("BATCH_RENDER_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}

# 平台审核意见 API（内网）。同步拒审判例，供语义审核参考，避免基线过时。
AUDIT_FEED_ENABLED = os.environ.get("AUDIT_FEED_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
AUDIT_FEED_BASE_URL = os.environ.get("AUDIT_FEED_BASE_URL", "http://192.168.31.76:8011").rstrip("/")
AUDIT_FEED_TIMEOUT_SECONDS = float(os.environ.get("AUDIT_FEED_TIMEOUT_SECONDS", "20"))
AUDIT_FEED_PAGE_SIZE = int(os.environ.get("AUDIT_FEED_PAGE_SIZE", "200"))
AUDIT_FEED_MAX_PAGES = int(os.environ.get("AUDIT_FEED_MAX_PAGES", "30"))
AUDIT_FEED_SYNC_HOURS = float(os.environ.get("AUDIT_FEED_SYNC_HOURS", "12"))
AUDIT_FEED_PROMPT_CASES = int(os.environ.get("AUDIT_FEED_PROMPT_CASES", "8"))

DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs").strip()
DEMUCS_DEVICE = os.environ.get("DEMUCS_DEVICE", "cuda").strip()
DEMUCS_TIMEOUT_SECONDS = float(os.environ.get("DEMUCS_TIMEOUT_SECONDS", "900"))

# 短镜补齐：只有 ffmpeg scene 分数 <= 该值的切点才允许合并成一个 Wan 单元。
# 实测这批素材切点分数在 0.34–0.94，默认 0.40 只放行很弱的切点。
MERGE_SCENE_SCORE_MAX = float(os.environ.get("MERGE_SCENE_SCORE_MAX", "0.40"))
WAN_MERGE_MAX_SECONDS = float(os.environ.get("WAN_MERGE_MAX_SECONDS", "4.0"))

_demucs_python_hint = os.environ.get("DEMUCS_PYTHON", "").strip()
if _demucs_python_hint and Path(_demucs_python_hint).is_file():
    DEMUCS_PYTHON: Path | None = Path(_demucs_python_hint)
else:
    _demucs_candidate = DATA_DIR / "tools" / "demucs-venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    DEMUCS_PYTHON = _demucs_candidate if _demucs_candidate.is_file() else None

STEMS_DIR = Path(os.environ.get("VIDEO_PRECHECK_STEMS_DIR") or DATA_DIR / "stems")

_ffmpeg_hint = os.environ.get("FFMPEG_PATH", "").strip()
if _ffmpeg_hint and Path(_ffmpeg_hint).is_file():
    FFMPEG_PATH = Path(_ffmpeg_hint)
else:
    _found_ffmpeg = shutil.which("ffmpeg")
    FFMPEG_PATH = Path(_found_ffmpeg) if _found_ffmpeg else None

_ffprobe_hint = os.environ.get("FFPROBE_PATH", "").strip()
if _ffprobe_hint and Path(_ffprobe_hint).is_file():
    FFPROBE_PATH = Path(_ffprobe_hint)
elif FFMPEG_PATH is not None:
    _candidate = FFMPEG_PATH.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    FFPROBE_PATH = _candidate if _candidate.is_file() else None
else:
    FFPROBE_PATH = None


def dashscope_root(base_url: str) -> str:
    """Normalize a DashScope base URL and drop API suffixes to avoid double paths."""
    root = base_url.strip().rstrip("/")
    for suffix in ("/compatible-mode/v1", "/api/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root


DASHSCOPE_ASR_ENDPOINT = (
    f"{dashscope_root(DASHSCOPE_ASR_BASE_URL)}"
    "/api/v1/services/aigc/multimodal-generation/generation"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STEMS_DIR.mkdir(parents=True, exist_ok=True)

# 口播语速口径：估算与文案预算统一按 5.0 字/秒，硬上限 5.8 字/秒（超过即拒绝生成）。
SPEECH_CHARS_PER_SECOND = float(os.environ.get("SPEECH_CHARS_PER_SECOND", "5.0"))
SPEECH_MAX_CHARS_PER_SECOND = float(os.environ.get("SPEECH_MAX_CHARS_PER_SECOND", "5.8"))
