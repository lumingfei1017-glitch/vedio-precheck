"""Wan 3.0 整镜重生成的薄客户端。

约束来自 2026-11-21 的 10 次实测：
- 输出时长不能超过参考片段时长，否则开始复读或发糊；
- 参考音频必须来自同一场景，否则画面会被带偏；
- 文案按 5–5.5 字/秒配，空档会被原句碎片填空。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable

import httpx

from .config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    SPEECH_CHARS_PER_SECOND,
    SPEECH_MAX_CHARS_PER_SECOND,
    dashscope_root,
)

WAN_MODEL = "wan3.0-video"
POLL_INTERVAL_SECONDS = 15
POLL_TIMEOUT_SECONDS = 2400
CHARS_PER_SECOND = SPEECH_CHARS_PER_SECOND
# 实测 5.5 字/秒是舒适上限，5.8 以上开始赶字；超过这个密度直接拒绝提交。
MAX_WAN_CHARS_PER_SECOND = SPEECH_MAX_CHARS_PER_SECOND
# 向上取整生成时允许的超长比例；超过就退回 floor，避免重演 G 版发糊。
WAN_DURATION_ROUND_UP_MAX_RATIO = 0.06

PROMPT_TEMPLATE = """视频1：原片片段，把这一段的口播整段改掉，只保留这一个镜头。
重要：删除视频1里原有的口播声音，不要沿用原台词，不要复读原句，必须逐字说下面这段新台词。
音频1：同一场景、同一人物的原声参考，只用于约束音色、语气和语速。
场景：{scene}
人物：视频1中的女性，发型、妆容、服装、首饰、产品与视频1完全一致。
动作：{action}
对白：
"{line}"
Lip sync：人物按照对白逐字说话，口型与台词严格同步。
约束：整段只保留视频1这一个镜头，禁止出现第二个场景、第二个人物、前后对比画面、产品特写；
禁止新增任何文字、字幕条、logo、水印；保持人物身份、服装、发型、妆容、产品、背景、镜头运动、光线一致；
不要新增背景音乐或音效；输出时长与参考视频一致。"""


class WanError(RuntimeError):
    """Wan 调用失败。"""


def char_budget(ref_seconds: float, cps: float = CHARS_PER_SECOND) -> int:
    """按参考时长给出文案字数上限。"""
    return max(4, int(ref_seconds * cps))


def planned_duration(
    ref_seconds: float,
    *,
    min_seconds: int = 2,
    max_seconds: int = 30,
) -> int:
    """输出时长取接近参考时长的整数秒。

    默认 floor：实测输出比参考长 28% 时会发糊吞字，所以不主动加长。
    但小数尾巴很大时（如 4.92s），floor 会白丢近 1 秒的口播容量；此时允许
    向上取整，代价只有 1–6% 的超长，生成后再 trim 回槽位。
    """
    target = int(ref_seconds)
    fraction = ref_seconds - target
    if fraction > 0 and ref_seconds > 0:
        overshoot = (1 - fraction) / ref_seconds
        if overshoot <= WAN_DURATION_ROUND_UP_MAX_RATIO:
            target += 1
    return max(min_seconds, min(max_seconds, target))


def text_fit(chars: int, seconds: int, *, low: float = 4.5, high: float = 6.5) -> str:
    """按目标时长评估文案密度：太短会被填空，太长会赶。"""
    if seconds <= 0:
        return "long"
    rate = chars / seconds
    if rate < low:
        return "short"
    if rate > high:
        return "long"
    return "ok"


def build_shot_prompt(
    line: str,
    *,
    scene: str = "室内竖屏服装展示口播",
    action: str = "她面向镜头站立介绍身上的产品，手部动作自然，镜头景别保持不变",
) -> str:
    if not line.strip():
        raise WanError("缺少新台词")
    return PROMPT_TEMPLATE.format(scene=scene, action=action, line=line.strip())


def _client(client: httpx.Client | None) -> httpx.Client:
    return client or httpx.Client(timeout=300)


def upload_media(
    path: Path,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = WAN_MODEL,
    client: httpx.Client | None = None,
) -> str:
    key = (api_key or DASHSCOPE_API_KEY).strip()
    if not key:
        raise WanError("未配置 DASHSCOPE_API_KEY")
    root = dashscope_root(base_url or DASHSCOPE_BASE_URL)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(path).name) or "input.bin"
    http = _client(client)
    try:
        policy = http.get(
            f"{root}/api/v1/uploads",
            headers={"Authorization": f"Bearer {key}"},
            params={"action": "getPolicy", "model": model},
        )
        policy.raise_for_status()
        data = policy.json()["data"]
        object_key = f"{data['upload_dir']}/{uuid.uuid4().hex[:8]}_{safe}"
        limit = int(data.get("max_file_size_mb") or 0)
        size = Path(path).stat().st_size
        if limit and size > limit * 1024 * 1024:
            raise WanError(f"素材超过 {limit}MB 上限")
        response = http.post(
            data["upload_host"],
            data={
                "OSSAccessKeyId": data["oss_access_key_id"],
                "policy": data["policy"],
                "Signature": data["signature"],
                "key": object_key,
                "x-oss-object-acl": data["x_oss_object_acl"],
                "x-oss-forbid-overwrite": data["x_oss_forbid_overwrite"],
                "success_action_status": "200",
            },
            files={"file": (safe, Path(path).read_bytes(), "application/octet-stream")},
        )
        response.raise_for_status()
    finally:
        if client is None:
            http.close()
    return f"oss://{object_key}"


def submit_task(
    media: list[dict],
    prompt: str,
    duration: int,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    resolution: str = "720P",
    model: str = WAN_MODEL,
    client: httpx.Client | None = None,
) -> str:
    key = (api_key or DASHSCOPE_API_KEY).strip()
    if not key:
        raise WanError("未配置 DASHSCOPE_API_KEY")
    if not media and not prompt.strip():
        raise WanError("prompt 与 media 不能同时为空")
    root = dashscope_root(base_url or DASHSCOPE_BASE_URL)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    if any(str(item.get("url", "")).startswith("oss://") for item in media):
        headers["X-DashScope-OssResourceResolve"] = "enable"
    payload = {
        "model": model,
        "input": {"prompt": prompt, "media": media},
        "parameters": {
            "resolution": resolution,
            "ratio": "adaptive",
            "duration": duration,
            "audio": True,
            "watermark": False,
            "seed": -1,
            "prompt_extend": False,
        },
    }
    http = _client(client)
    try:
        response = http.post(
            f"{root}/api/v1/services/aigc/video-generation/video-synthesis",
            headers=headers,
            json=payload,
        )
    finally:
        if client is None:
            http.close()
    body = response.json()
    if response.status_code >= 400:
        raise WanError(f"提交失败 {response.status_code}: {json.dumps(body, ensure_ascii=False)[:400]}")
    task_id = (body.get("output") or {}).get("task_id")
    if not task_id:
        raise WanError(f"提交未返回 task_id: {json.dumps(body, ensure_ascii=False)[:400]}")
    return task_id


def poll_task(
    task_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    interval: float = POLL_INTERVAL_SECONDS,
    timeout: float = POLL_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    client: httpx.Client | None = None,
) -> dict:
    key = (api_key or DASHSCOPE_API_KEY).strip()
    root = dashscope_root(base_url or DASHSCOPE_BASE_URL)
    deadline = time.time() + timeout
    http = _client(client)
    try:
        while time.time() < deadline:
            response = http.get(f"{root}/api/v1/tasks/{task_id}", headers={"Authorization": f"Bearer {key}"})
            body = response.json()
            output = body.get("output") or {}
            status = output.get("task_status", "")
            if status == "SUCCEEDED":
                if not output.get("video_url"):
                    raise WanError("任务成功但缺少 video_url")
                return body
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                raise WanError(f"任务 {status}: {output.get('message') or output.get('code')}")
            sleep(interval)
    finally:
        if client is None:
            http.close()
    raise WanError("任务轮询超时")


def download_result(url: str, dest: Path, *, client: httpx.Client | None = None) -> Path:
    if ".aliyuncs.com" not in url:
        raise WanError("结果地址不在受信域名内")
    http = _client(client)
    try:
        response = http.get(url, follow_redirects=True)
        response.raise_for_status()
    finally:
        if client is None:
            http.close()
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_bytes(response.content)
    return Path(dest)
