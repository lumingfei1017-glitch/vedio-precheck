# 后台审核任务

## 目的

单条素材的 ASR 需要 10-45 秒，语义审核需要 10-40 秒。同步 HTTP 连接不适合 6-8 人批量使用，因此上传和审核都支持后台任务。

## API

### 后台上传

`POST /api/batches/{batch_id}/materials` 的 multipart 表单增加：

```text
background=true
```

返回 `202 Accepted`：

```json
{
  "id": "material-id",
  "status": "queued",
  "job_id": "job-id"
}
```

任务会按需执行 ASR、规则审核和语义审核。

### 后台单条审核

```text
POST /api/materials/{material_id}/review?background=true
```

### 后台批量审核

```text
POST /api/batches/{batch_id}/review?background=true
```

### 查询任务

```text
GET /api/jobs/{job_id}
GET /api/jobs?limit=20
```

任务字段：

- `status`: `queued | running | succeeded | partial | failed`
- `stage`: 当前阶段（识别口播、规则审核、语义审核、视觉审核、上下文改写、合并证据、保存结果）
- `progress`: 0-100
- `result`: 完成后的素材/批次统计
- `error`: 失败原因

## 实现

- `app/jobs.py`：SQLite 任务表 + 可重启的 `ThreadPoolExecutor`（默认 2 worker）。
- 上传和审核接口通过 `background=true` 入队，未传时保留同步行为，兼容现有调用。
- 工作台显示进度条，每 1.5 秒轮询一次。
- 服务启动时把上次未完成的任务标为失败，避免任务永久卡在 running。
- `kind=render_material` 的渲染任务阶段：准备素材、剪辑拼接、生成配音、音频合成、字幕、完成。
