# video-precheck

千川投流素材的**违规审核 + 一键改片**工作台（内网 6–8 人使用）。

上传视频后自动识别口播，产出带时间码的问题、建议文案、A/B/C 修改方式、复检结果和人工修改清单。
左侧完整口播文案会跟随视频播放逐句变绿，点任意句可跳转播放。

- 文案类问题：规则快筛 + 语义 Agent 跨句聚合 → Wan 单镜口播改写
- 画面类问题：关键帧 + VLM 判定 → 导出人工清单到剪映
- 当前自动化的边界：**只自动做 Wan 纯口播改写**；TTS 局部换音与自动删段默认关闭，一律转人工

## 快速开始

需要 Python 3.12（见 `.python-version`）与 FFmpeg。

```powershell
uv sync                                  # 建/同步 .venv（已锁定 uv.lock）
Copy-Item .env.example .env              # 填 DASHSCOPE_API_KEY 和 FFMPEG_PATH
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

打开 http://127.0.0.1:8000

uv 不可用时直接跑已有 venv 也一样：

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

环境自检：

```powershell
uv sync --check                # 应输出 "Would make no changes"
.venv\Scripts\python.exe -m pytest -q
```

## 配置

所有配置走 `.env`（参考 `.env.example`，真实 `.env` 已 gitignore）。
已存在的系统环境变量优先，不会被 `.env` 覆盖。

| 变量 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | **必填**，ASR / 语义 / 视觉 / 改写 / TTS 共用 |
| `BASE_URL1` | OpenAI 兼容 v1 接口，默认 DashScope compatible-mode |
| `FFMPEG_PATH` / `FFPROBE_PATH` | 不填则从 PATH 查找 |
| `SEMANTIC_ENABLED` / `SEMANTIC_MODEL` | 语义审核开关与模型 |
| `VISUAL_ENABLED` / `VISUAL_MODEL` | 视觉审核开关与模型 |
| `REWRITE_ENABLED` / `REWRITE_MODEL` | 改写 Agent 开关与模型 |
| `TTS_ROUTE_ENABLED` | 局部换音，**默认 false** |
| `AUTO_CUT_ENABLED` | 自动删段，**默认 false** |
| `BATCH_RENDER_ENABLED` | 批量 Wan 渲染，默认 true；关闭后 `/apply?dry_run=false` 返回 409 |
| `AUDIT_FEED_ENABLED` / `AUDIT_FEED_BASE_URL` | 平台审核意见内网只读接口 |
| `DEMUCS_PYTHON` | Demucs 外部解释器路径 |

`.env` 加载器是项目内自写的简单 `KEY=VALUE` 解析（`app/config.py`），
支持注释、引号、UTF-8 BOM，不展开变量。

## 目录结构

```text
app/
  main.py         FastAPI 启动、静态资源、启动时后台同步审核意见
  api.py          所有 /api 路由
  config.py       环境变量解析与路径常量
  db.py           SQLite 建表（batch/material/finding/suggestion/recheck/feedback/job/audit_*）
  jobs.py         后台任务队列（ThreadPoolExecutor，2 worker）
  asr.py          DashScope ASR：抽 16k 单声道 → Data URL → 词级时间戳 → SRT
  srt.py          SRT 解析 / 渲染 / 文本互转
  rules.py        规则快筛（rules/qianchuan_apparel.json，14 条规则，版本 0.2）
  semantic.py     语义事件审核（跨句聚合，事件级 finding + 多条 evidence）
  visual.py       关键帧抽取 + VLM 视觉审核 + 事件密集复核
  review.py       审核编排、结果落库、复检
  suggest.py      建议草稿与文案刷新
  rewrite.py      上下文改写 Agent（带文案容量约束）
  edit.py         修改计划（A/B/C 分类、删段、路由对齐）
  boundary.py     把 ASR 边界吸附到真实静音处
  separate.py     Demucs 人声/BGM 分离（外部解释器，按内容 hash 缓存）
  shots.py        镜头切分 + 类型判定 + 路由（wan/tts/keep/manual）
  wan.py          Wan 3.0 薄客户端（时长、文案密度、首尾复读约束）
  replace.py      整镜替换服务层（切参考 → 上传 → 生成 → 回贴）
  worklist.py     人工修改清单（Markdown + 可导入剪映的 SRT 标记轨）
  audit_feed.py   平台审核意见同步与口径统计
  rule_miner.py   从判例 n-gram 挖候选规则
  tts.py          CosyVoice 克隆优先、Edge 兜底（默认不进主链路）
scripts/          评测、探针、离线 CLI（见下方"脚本"）
rules/            规则 JSON、候选词、提示词锚点
static/           单页工作台（无构建步骤）
tests/            pytest，离线运行，不需要 API Key
docs/             设计与评测文档
```

## 审核链路

1. **ASR**：FFmpeg 抽 16kHz 单声道 MP3 → Base64 Data URL → `qwen-audio-3.0-asr-flash`，得到词级时间戳。
   也支持上传 SRT 或直接粘贴口播稿，文本优先于 ASR。3 分钟素材实测约 43 秒。
2. **规则快筛**：14 条规则保显性词和高置信兜底（`rules/qianchuan_apparel.json`）。
3. **语义审核**：对完整口播做跨句事件聚合，输出事件级 finding，每条带多条 evidence（原文 + 时间码）。
   被事件覆盖的规则 finding 自动去重；调用失败时回退规则结果并记录 `semantic_status`。
4. **视觉审核**：均匀抽帧（最多 16 帧，512px）判定低俗擦边、品牌混淆、极端拉伸、穿前穿后对比。
   命中后对每个视觉事件做一次密集复核（±12s、1s 步长），时间码会精确到秒级。
5. **改写**：读取完整口播 + 前后句 + 时间码，把违规句改成可直接口播的成稿，输出主文案 + 1 条备选。
6. **复检 / 清单**：产出可下载的人工清单（Markdown）和 SRT 标记轨。

### 评测基线

| 口径 | 结果 |
| --- | --- |
| 42 条文本类视频（91 条平台引文）部分命中 | 规则 80/91 → 合并 **86/91（94.5%）** |
| 同上，完整命中 | 规则 61/91 → 合并 **80/91（87.9%）** |
| 24 条纯画面视频，类别覆盖 | 23/41；类别 + 画面关键词 17/41（41.5%） |

细节见 [docs/semantic-eval.md](docs/semantic-eval.md)、[docs/visual-eval.md](docs/visual-eval.md)。

## 文案时长硬约束

Wan 生成时长不能超过参考镜头，否则会复读或发糊。三层拦截：

1. 改写提示词按 target 给出 `max_chars = 时长 × 5.0 字/秒`，要求绝不超；
2. 超预算自动再压一次（strict 模式）；
3. `plan_replace()` 兜底：超过 `时长 × 5.8 字/秒` 直接拒绝提交。

语速口径统一按 **5.0 字/秒**，硬拒绝阈值 **5.8 字/秒**（`SPEECH_CHARS_PER_SECOND` /
`SPEECH_MAX_CHARS_PER_SECOND`）。同时限制缩字不超过 20%，避免丢卖点。

## 镜头与路由

`app/shots.py` 先把素材拆成镜头清单，再按"镜头时长 + 嘴是否可见 + 违规覆盖率"决定处理方式：

| 镜头类型 | 含义 | 路由 |
| --- | --- | --- |
| `talking_closeup` | 脸占比 ≥6%，嘴清晰 | `wan` |
| `talking_full` | 人脸持续可见但较小 | `wan` |
| `talking_side` | 侧身/转身 | `wan`（需 ≥2s） |
| `voiceover` | 有人声无人脸 | `tts`（当前转人工） |
| `broll` | 无口播 | `keep` |
| `unknown` | 没装 cv2 或待确认 | `manual` |

短镜补齐：<2s 的替换区间会并入相邻口播镜到 2–4s，但必须先过 `scene <= MERGE_SCENE_SCORE_MAX`
（默认 0.40）的画面闸门。实测只有约 7% 的切点够软。

Wan 只接收 2–30 整数秒，所以回贴必须等长：短于槽位的片段用末帧冻结 + 静音补齐，再 trim 到槽位。

## Wan 渲染的硬规则

来自 10 次实测（详见 [docs/plan-audio-block-replace.md](docs/plan-audio-block-replace.md)）：

1. 生成时长 ≤ 参考镜头时长，取 `floor()`；
2. 参考音频必须同场景，否则会把别的场景带进画面；
3. 文案按 5–5.5 字/秒，太短会被原句碎片填空，太满会赶；
4. 新文案首尾 5 字不要与原句相同，否则会复读；
5. 替换粒度是整镜，切点必须落在原片硬切上。

批量执行时逐镜判定（`_wan_preflight`），不合格直接跳过并保留原样，其余继续，
最终返回 `applied` / `skipped`（含原因）。

## 人工清单

工作台只自动执行 Wan 纯口播改写，其余问题导出为清单人工处理：

```text
GET /api/materials/{id}/worklist.md    # 可直接带到剪映的工单
GET /api/materials/{id}/worklist.srt   # 可导入剪映的标记轨
```

时间码基于原始视频，不删段、不重排，可在剪映里直接跳转。

## API

```text
GET  /api/health
GET  /api/batches                     POST /api/batches
POST /api/batches/{id}/materials      GET  /api/materials?batch_id=
GET  /api/materials/{id}              POST /api/materials/{id}/review?background=
POST /api/batches/{id}/review         POST /api/materials/{id}/transcribe
POST /api/suggestions/{id}/refresh    POST /api/suggestions/{id}/feedback
POST /api/materials/{id}/confirm      POST /api/materials/{id}/apply?dry_run=
GET  /api/materials/{id}/shots?refresh=false
POST /api/materials/{id}/shots/replace
GET  /api/jobs                        GET  /api/jobs/{job_id}
POST /api/audits/sync                 GET  /api/audits/stats
GET  /api/materials/{id}/output       GET  /api/materials/{id}/output.srt
GET  /api/materials/{id}/video        GET  /api/materials/{id}/srt
GET  /api/materials/{id}/report.json
GET  /api/materials/{id}/worklist.md  GET  /api/materials/{id}/worklist.srt
```

后台任务：上传、单条审核和批量审核都支持 `background=true`，立即返回 `job_id`。
阶段包括排队、识别口播、规则审核、语义审核、合并证据、保存结果；任务表在 `job`。
线程池默认 2 个 worker。

`/shots/replace` 的 `dry_run=true` 只返回 `duration / char_budget / fit / warnings`，不花钱；
`dry_run=false` 返回 `job_id`（`kind=shot_replace`），复用 `/api/jobs/{id}` 轮询。

## 脚本

```powershell
# 平台审核意见同步（服务启动时也会按 AUDIT_FEED_SYNC_HOURS 自动同步）
.venv\Scripts\python.exe scripts\sync_audits.py

# 从判例挖候选关键词规则 → rules/candidates.json
.venv\Scripts\python.exe scripts\mine_rules.py

# 单独验证 ASR
.venv\Scripts\python.exe scripts\asr_smoke.py data\demo.mp4

# 参考表 / 参考视频回归评测
.venv\Scripts\python.exe scripts\eval_reference.py
.venv\Scripts\python.exe scripts\ref_eval.py --text-only --limit 42

# 整镜替换离线脚本（--dry-run 只切参考片段，不调用接口）
python scripts\replace_shot.py --video input.mp4 --start 78.2 --end 83.7 `
  --text "这款平时穿也合适，喜欢的姐妹可以试试看。" --out output.mp4
```

其它：`semantic_eval.py`、`visual_eval.py`、`llm_smoke.py`、`vision_smoke.py`、`semantic_smoke.py`、
`cache_all_asr.py`、`prepare_violation_buckets.py`、`analyze_ref_eval.py`、`report_ref_results.py`。

## 判例反哺

平台内网只读接口每天凌晨更新千川全账户的拒审 / 待优化意见，比静态 Excel 更适合做口径基线。

- 同步链路：分页拉取 → 拆成「意见模板 / 括号明细 / 违规原句」→ upsert 到 `audit_case`。
- 口径参考：取最高频的 8 类口径 + 近期原句注入语义审核提示词，让判定跟得上平台变化。
- **只作口径参考，不自动生成阻断规则**。新口径要变成硬规则仍需人工确认，避免误报扩散。

首次同步（2026-11-24）：1061 条拒审记录 → 1321 条 reason → 34 类口径，
其中「夸大功效」878 条、「不良导向」154 条、「内衣入口图真人素材」83 条。

判例挖出的高精度候选（全为 0 通过样本）：`显瘦十几斤`（65 素材）、`收腹又提臀`（49）、
`轻松显瘦十几斤`（26）。`紧致` / `显瘦` / `提拉` 覆盖广但太通用，只宜做 suggest。

## 环境说明

- 项目自带 `.venv`（Python 3.12.11）已与 `uv.lock` 同步，`uv sync --check` 无变更。
- `opencv-python-headless` 锁 `<5`：OpenCV 5 移除了 `CascadeClassifier`，装上会让镜头全判 `unknown`。
- `dashscope` SDK 只在音色克隆探针里用，主链路走 HTTP，不依赖它。
- Demucs 依赖 torch，不放进主依赖，需要单独准备解释器：

```ini
DEMUCS_PYTHON=data\tools\demucs-venv\Scripts\python.exe
DEMUCS_MODEL=htdemucs
DEMUCS_DEVICE=cuda
```

## 已知限制

1. **TTS 音质不达标**，局部换音与自动删段已停用，短镜（<2s）、无脸/侧身镜和人声分离后的
   BGM 保留场景全部转人工。
2. **硬字幕无法擦除**，需要 OCR 定位 + 重绘。
3. **画面类召回仍低**（类别+关键词 41.5%），视觉结果只作候选和证据链，必须人工确认。
4. **渲染后未自动复检**：缺 ASR 逐字比对、与参考音频相关度、频谱重心这几项自动判定。
5. **任务队列未隔离**：Wan 会占住 2 个审核线程之一，批量时审核任务会被拖慢。
6. **跨镜头主张未聚合**：一条违规主张可能横跨口播 + 对比镜头，目前按单镜头处理。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/handoff-2026-11-21.md](docs/handoff-2026-11-21.md) | 完整交接：M1–M4 状态、实测数据、踩坑速查 |
| [docs/decision-wan-first-routing.md](docs/decision-wan-first-routing.md) | 路由 v2 判定依据 |
| [docs/plan-audio-block-replace.md](docs/plan-audio-block-replace.md) | 片段替换方案与 Wan 实测约束 |
| [docs/semantic-review.md](docs/semantic-review.md) / [docs/semantic-eval.md](docs/semantic-eval.md) | 语义审核与评测 |
| [docs/visual-review.md](docs/visual-review.md) / [docs/visual-eval.md](docs/visual-eval.md) | 视觉审核与评测 |
| [docs/rewrite-agent.md](docs/rewrite-agent.md) | 改写 Agent |
| [docs/video-modification.md](docs/video-modification.md) | 视频修改 MVP |
| [docs/background-jobs.md](docs/background-jobs.md) | 后台任务 |
| [docs/ref-eval-summary.md](docs/ref-eval-summary.md) / [docs/ref-eval-pilot.md](docs/ref-eval-pilot.md) | 参考视频评测 |
| [docs/plan-speaker-voice-profile.md](docs/plan-speaker-voice-profile.md) / [docs/spike-speaker-diarization.md](docs/spike-speaker-diarization.md) | 音色档案与说话人分离 |

## 测试

```powershell
.venv\Scripts\python.exe -m pytest -q
```

132 个测试，全部离线运行（`tests/conftest.py` 把数据目录指向临时目录，并关闭语义/视觉/改写/
审核意见同步），不需要 API Key 和网络。