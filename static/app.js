const state = {
  batches: [],
  materials: [],
  report: null,
  selectedMaterialId: null,
  transcriptRows: [],
  activeTranscriptIndex: null,
  activeJobId: null,
  jobPollTimer: null,
  jobContext: null,
  shots: [],
  shotsMaterialId: null,
  planMap: {},
  planWanCount: 0,
};

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmtTime(ms) {
  if (ms === null || ms === undefined) return "未定位";
  const total = Math.max(0, Number(ms)) / 1000;
  const minutes = Math.floor(total / 60);
  const seconds = (total % 60).toFixed(1).padStart(4, "0");
  return `${String(minutes).padStart(2, "0")}:${seconds}`;
}

function fmtDuration(ms) {
  if (ms === null || ms === undefined) return "-";
  return `${(Number(ms) / 1000).toFixed(1)}s`;
}

function severityLabel(value) {
  return { blocking: "阻断", suggest: "建议改", info: "提示" }[value] || value;
}

function methodLabel(value) {
  return {
    TTS_SUBTITLE: "人工配音（TTS 已停用）",
    CUT_BROLL: "删句/换镜头",
    WAN_LIPSYNC: "Wan 口型",
    MANUAL: "人工处理",
  }[value] || value;
}

const SELECTABLE_METHODS = ["WAN_LIPSYNC", "CUT_BROLL", "MANUAL"];

function methodChoices(current) {
  const options = SELECTABLE_METHODS.slice();
  if (current && !options.includes(current)) options.unshift(current);
  return options
    .map((method) => {
      const legacy = current === method && !SELECTABLE_METHODS.includes(method);
      const selected = current === method ? " selected" : "";
      const disabled = legacy ? " disabled" : "";
      return `<option value="${method}"${selected}${disabled}>${methodLabel(method)}</option>`;
    })
    .join("");
}

function classLabel(value) {
  return { A: "A 无正脸", B: "B 可删句", C: "C 保留口播", MANUAL: "需人工" }[value] || value;
}

function statusLabel(value) {
  return {
    uploaded: "待审核",
    queued: "排队中",
    transcribing: "识别中",
    asr_failed: "识别失败",
    reviewed: "已审核",
    confirmed: "已确认",
  }[value] || value;
}

function sourceLabel(value) {
  return { srt: "SRT", script: "口播稿", asr: "自动识别" }[value] || "未识别";
}

function categoryLabel(value) {
  return value === "shapewear" ? "塑身衣" : value || "塑身衣";
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

let toastTimer = null;
function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 3200);
}

function jobStageLabel(stage) {
  const value = String(stage || "");
  const labels = [
    ["识别口播", "识别口播"],
    ["rules", "规则审核"],
    ["semantic", "语义审核"],
    ["visual", "视觉审核"],
    ["rewrite", "上下文改写"],
    ["merge", "合并证据"],
    ["保存结果", "保存结果"],
    ["queued", "排队中"],
    ["准备", "准备中"],
    ["prepare", "准备中"],
    ["切参考片段", "切参考片段"],
    ["上传素材", "上传素材"],
    ["提交生成任务", "提交生成任务"],
    ["等待 Wan 生成", "Wan 生成中"],
    ["回贴合成", "回贴合成"],
    ["batch", "批量处理"],
    ["done", "完成"],
    ["interrupted", "已中断"],
    ["finished_with_errors", "部分完成"],
  ];
  for (const [key, label] of labels) {
    if (value.includes(key)) return value.replace(key, label);
  }
  return value || "处理中";
}

function renderJob(job) {
  const banner = $("#jobBanner");
  const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
  banner.classList.remove("hidden");
  banner.classList.toggle("failed", job.status === "failed");
  $("#jobStage").textContent = jobStageLabel(job.stage);
  $("#jobPercent").textContent = `${Math.round(progress)}%`;
  $("#jobBar").style.width = `${progress}%`;
}

function stopJobPolling() {
  if (state.jobPollTimer) clearInterval(state.jobPollTimer);
  state.jobPollTimer = null;
  state.activeJobId = null;
}

function startJobPolling(jobId, context = {}) {
  stopJobPolling();
  state.activeJobId = jobId;
  state.jobContext = context;
  renderJob({ status: "queued", stage: "queued", progress: 0 });

  const tick = async () => {
    if (!state.activeJobId) return;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      renderJob(job);
      if (job.status === "succeeded" || job.status === "partial") {
        const finished = { ...job, progress: 100, stage: "done" };
        stopJobPolling();
        renderJob(finished);
        await loadMaterials();
        const materialId = job.material_id || job.result?.material_id || context.materialId;
        const batchId = job.batch_id || job.result?.batch_id || context.batchId;
        if (materialId) {
          await openMaterial(materialId);
        } else if (batchId && state.materials.length) {
          await openMaterial(state.materials[0].id);
        }
        if (job.status === "partial") {
          toast(`批量审核完成，${job.result?.failed?.length || 0} 条失败`, true);
        } else if (job.kind === "shot_replace" && job.result?.fallback_from === "wan") {
          toast("Wan 重生成失败，该镜头请转人工处理", true);
        } else if (job.kind === "shot_replace" && job.result?.voice_source === "edge_fallback") {
          toast("换音轨完成，但使用了 Edge 测试音色", true);
        } else if (job.kind === "render_material") {
          const skipped = (job.result && job.result.skipped) || [];
          if (skipped.length) {
            toast(`修改版已生成，${skipped.length} 段未满足 Wan 规则，已保留原样`, true);
          } else {
            toast("修改版已生成，可下载");
          }
        } else if (job.kind === "shot_replace") {
          toast("修改版已生成，可下载");
        } else {
          toast("审核完成");
        }
        setTimeout(() => $("#jobBanner").classList.add("hidden"), 1800);
      } else if (job.status === "failed") {
        stopJobPolling();
        renderJob({ ...job, progress: 100 });
        toast(job.error || "审核任务失败", true);
      }
    } catch (error) {
      toast(error.message, true);
    }
  };

  tick();
  state.jobPollTimer = setInterval(tick, 1500);
}

async function loadBatches() {
  state.batches = await api("/api/batches");
  const select = $("#batchSelect");
  select.innerHTML = "";
  if (!state.batches.length) {
    select.innerHTML = '<option value="">暂无批次</option>';
    $("#batchForm").classList.remove("hidden");
    $("#materialList").innerHTML = "";
    return;
  }
  $("#batchForm").classList.add("hidden");
  for (const batch of state.batches) {
    const option = document.createElement("option");
    option.value = batch.id;
    option.textContent = `${batch.name} (${batch.material_count})`;
    select.appendChild(option);
  }
  await loadMaterials();
}

async function loadMaterials() {
  const batchId = $("#batchSelect").value;
  if (!batchId) return;
  state.materials = await api(`/api/materials?batch_id=${encodeURIComponent(batchId)}`);
  const list = $("#materialList");
  list.innerHTML = "";
  if (!state.materials.length) {
    list.innerHTML = '<p class="muted">这个批次还没有素材。</p>';
    return;
  }
  for (const material of state.materials) {
    const item = document.createElement("div");
    item.className = `material-item${material.id === state.selectedMaterialId ? " active" : ""}`;
    item.dataset.materialId = material.id;
    item.innerHTML = `
      <div class="name">${escapeHtml(material.filename)}</div>
      <div class="meta">
        <span>${escapeHtml(statusLabel(material.status))}</span>
        <span>${material.finding_count || 0} 个问题</span>
      </div>
    `;
    item.addEventListener("click", () => openMaterial(material.id));
    list.appendChild(item);
  }
}

async function createBatch(event) {
  event.preventDefault();
  const name = $("#batchName").value.trim();
  if (!name) return;
  await api("/api/batches", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  $("#batchName").value = "";
  $("#batchForm").classList.add("hidden");
  toast("批次已创建");
  await loadBatches();
}

async function uploadMaterial(event) {
  event.preventDefault();
  const batchId = $("#batchSelect").value;
  if (!batchId) {
    toast("请先创建或选择一个批次", true);
    return;
  }
  const video = $("#videoInput").files[0];
  if (!video) {
    toast("请选择视频文件", true);
    return;
  }
  const form = new FormData();
  form.append("video", video);
  const srt = $("#srtInput").files[0];
  if (srt) form.append("srt", srt);
  form.append("script", $("#scriptInput").value);
  form.append("has_face", $("#hasFaceSelect").value);
  form.append("background", "true");

  const button = $("#uploadBtn");
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "上传中…";
  try {
    const result = await api(`/api/batches/${encodeURIComponent(batchId)}/materials`, {
      method: "POST",
      body: form,
    });
    $("#uploadForm").reset();
    await loadMaterials();
    startJobPolling(result.job_id, { materialId: result.id, batchId });
    toast("素材已上传，正在识别和审核");
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

async function openMaterial(materialId) {
  if (state.shotsMaterialId !== materialId) {
    state.shots = [];
    state.shotsMaterialId = materialId;
  }
  state.selectedMaterialId = materialId;
  state.report = await api(`/api/materials/${encodeURIComponent(materialId)}`);
  state.planMap = await loadPlanMap(materialId);
  renderReport();
  await loadMaterials();
}

async function loadPlanMap(materialId) {
  try {
    const plan = await api(`/api/materials/${encodeURIComponent(materialId)}/apply?dry_run=true`, {
      method: "POST",
    });
    const map = {};
    let wanCount = 0;
    for (const op of plan.operations || []) {
      if (op.op !== "replace_script") continue;
      if (op.route === "wan") wanCount += 1;
      for (const findingId of op.finding_ids || []) {
        map[findingId] = {
          route: op.route || "manual",
          start_ms: op.start_ms,
          end_ms: op.end_ms,
          reason: op.route_reason || "",
        };
      }
    }
    state.planWanCount = wanCount;
    return map;
  } catch (error) {
    return {};
  }
}

function renderReport() {
  const report = state.report;
  if (!report) return;
  $("#emptyState").classList.add("hidden");
  $("#workspace").classList.remove("hidden");
  $("#materialTitle").textContent = report.material.filename;
  $("#materialMeta").textContent = [
    categoryLabel(report.material.category),
    `状态：${statusLabel(report.material.status)}`,
    `时长：${fmtDuration(report.material.duration_ms)}`,
    `字幕：${sourceLabel(report.material.transcript_source)}`,
    report.material.has_face === true ? "有正脸" : report.material.has_face === false ? "无正脸" : "正脸未确认",
  ].join(" · ");
  const player = $("#videoPlayer");
  if (player.dataset.materialId !== report.material.id) {
    player.dataset.materialId = report.material.id;
    player.src = `/api/materials/${encodeURIComponent(report.material.id)}/video`;
  }
  $("#downloadBtn").classList.toggle("hidden", report.material.render_status !== "succeeded");

  const summary = report.summary || {};
  $("#summary").innerHTML = `
    <span class="chip"><strong>${summary.total || 0}</strong> 个问题</span>
    <span class="chip red"><strong>${summary.blocking || 0}</strong> 阻断</span>
    <span class="chip amber"><strong>${summary.suggest || 0}</strong> 建议</span>
    <span class="chip"><strong>${summary.manual || 0}</strong> 需人工</span>
    ${summary.semantic_count ? `<span class="chip"><strong>${summary.semantic_count}</strong> 语义事件</span>` : ""}
    ${summary.visual_count ? `<span class="chip"><strong>${summary.visual_count}</strong> 视觉事件</span>` : ""}
    ${summary.rewrite_count ? `<span class="chip"><strong>${summary.rewrite_count}</strong> 已改写</span>` : ""}
    ${summary.semantic_status === "failed" ? `<span class="chip red" title="${escapeHtml(summary.semantic_error || "")}">语义审核失败</span>` : ""}
    ${summary.visual_status === "failed" ? `<span class="chip red" title="${escapeHtml(summary.visual_error || "")}">视觉审核失败</span>` : ""}
    ${summary.rewrite_status === "failed" ? `<span class="chip red" title="${escapeHtml(summary.rewrite_error || "")}">文案改写失败</span>` : ""}
    ${summary.render_status === "running" ? '<span class="chip amber">修改版生成中</span>' : ""}
    ${summary.render_status === "failed" ? `<span class="chip red" title="${escapeHtml(summary.render_error || "")}">修改版生成失败</span>` : ""}
    ${summary.render_status === "succeeded" ? '<span class="chip green">修改版已生成</span>' : ""}
    ${summary.false_positive ? `<span class="chip"><strong>${summary.false_positive}</strong> 已忽略</span>` : ""}
    <span class="chip"><strong>${summary.a_count || 0}</strong> A / <strong>${summary.b_count || 0}</strong> B / <strong>${summary.c_count || 0}</strong> C</span>
    <span class="chip">规则 v${escapeHtml(summary.rule_version || "-")}</span>
    <span class="chip">${escapeHtml(sourceLabel(summary.transcript_source))}${summary.word_count ? ` · ${summary.word_count} 词` : ""}</span>
  `;

  const asrBanner = $("#asrBanner");
  if (report.material.status === "asr_failed" || report.material.asr_error) {
    asrBanner.className = "recheck failed";
    asrBanner.textContent = `字幕识别失败：${report.material.asr_error || "未识别到人声"}。可点“识别字幕”重试，或重新上传 SRT。`;
  } else if (report.material.status === "transcribing") {
    asrBanner.className = "recheck";
    asrBanner.textContent = "正在识别口播并生成字幕，请稍候。";
  } else if (report.material.status === "queued") {
    asrBanner.className = "recheck";
    asrBanner.textContent = "任务已排队，正在等待识别和审核。";
  } else {
    asrBanner.className = "recheck hidden";
  }

  const recheck = report.recheck;
  const banner = $("#recheckBanner");
  if (recheck) {
    const passed = recheck.status === "passed";
    banner.className = `recheck ${passed ? "passed" : "failed"}`;
    banner.textContent = passed
      ? `复检通过：未发现新的词面阻断项。${recheck.manual_count ? `另有 ${recheck.manual_count} 项需人工处理。` : "仍需人工看语境和画面。"}`
      : `复检需修改：仍有 ${recheck.blocking_count || 0} 个阻断项。`;
  } else {
    banner.className = "recheck hidden";
  }

  $("#findingCount").textContent = `${report.findings.length} 条`;
  renderFindings(report.findings || []);
  renderTranscript(report);
  $("#planPanel").classList.add("hidden");
}

function renderTranscript(report) {
  const panel = $("#transcriptPanel");
  const body = $("#transcriptBody");
  const count = $("#transcriptCount");
  const segments = Array.isArray(report.segments) ? report.segments : [];
  let rows = segments.map((segment) => ({
    index: segment.index,
    start_ms: segment.start_ms,
    end_ms: segment.end_ms,
    text: segment.text,
  }));

  if (!rows.length) {
    rows = String(report.transcript || "")
      .split(/\n+/)
      .map((text) => text.trim())
      .filter(Boolean)
      .map((text, index) => ({ index: index + 1, start_ms: null, end_ms: null, text }));
  }

  state.transcriptRows = rows;
  state.activeTranscriptIndex = null;
  if (!rows.length) {
    panel.classList.add("hidden");
    body.innerHTML = "";
    return;
  }

  const flagged = new Map();
  for (const finding of report.findings || []) {
    if (finding.start_ms === null || finding.end_ms === null) continue;
    for (const row of rows) {
      if (row.start_ms === null || row.end_ms === null) continue;
      if (row.start_ms < finding.end_ms && row.end_ms > finding.start_ms) {
        if (!flagged.has(row.index)) flagged.set(row.index, new Set());
        flagged.get(row.index).add(finding.rule_id);
      }
    }
  }

  panel.classList.remove("hidden");
  count.textContent = segments.length ? `${rows.length} 句` : `${rows.length} 行`;
  body.innerHTML = rows.map((row) => {
    const rules = flagged.get(row.index);
    const timed = row.start_ms !== null && row.end_ms !== null;
    const attrs = timed
      ? `data-start="${row.start_ms}" data-end="${row.end_ms}"`
      : "disabled";
    return `
      <button type="button" class="transcript-line${rules ? " flagged" : ""}" data-index="${row.index}" ${attrs}>
        <span class="tc">${timed ? fmtTime(row.start_ms) : "—"}</span>
        <span class="txt">${escapeHtml(row.text)}${rules ? `<span class="transcript-rule">${Array.from(rules).join(" ")}</span>` : ""}</span>
      </button>
    `;
  }).join("");
  syncTranscriptToVideo();
}

function syncTranscriptToVideo() {
  const player = $("#videoPlayer");
  const timeMs = player.currentTime * 1000;
  const rows = document.querySelectorAll("#transcriptBody .transcript-line[data-start]");
  let active = null;
  for (const row of rows) {
    const start = Number(row.dataset.start);
    const end = Number(row.dataset.end);
    const isActive = timeMs >= start && timeMs < end;
    row.classList.toggle("active", isActive);
    if (isActive) active = row;
  }
  if (active && state.activeTranscriptIndex !== active.dataset.index) {
    state.activeTranscriptIndex = active.dataset.index;
    const body = $("#transcriptBody");
    const bodyRect = body.getBoundingClientRect();
    const rowRect = active.getBoundingClientRect();
    if (rowRect.top < bodyRect.top || rowRect.bottom > bodyRect.bottom) {
      body.scrollTop += rowRect.top - bodyRect.top - 8;
    }
  }
}

function seekTranscript(row) {
  const start = Number(row.dataset.start);
  if (!Number.isFinite(start)) return;
  const player = $("#videoPlayer");
  player.currentTime = start / 1000;
  player.play().catch(() => {});
  syncTranscriptToVideo();
}

function renderFindings(findings) {
  const container = $("#findings");
  container.innerHTML = "";
  if (!findings.length) {
    container.innerHTML = `
      <div class="empty" style="min-height:180px">
        <strong>未发现词面候选</strong>
        <span>零候选不等于安全，仍需按判定手册做全文语义复核。</span>
      </div>
    `;
    return;
  }

  for (const finding of findings) {
    const suggestion = finding.suggestion;
    const card = document.createElement("article");
    card.className = `finding ${finding.severity}${suggestion && suggestion.status === "false_positive" ? " rejected" : ""}`;
    card.dataset.findingId = finding.id;
    const durationText = suggestion
      ? `${fmtDuration(suggestion.original_duration_ms)} → ${fmtDuration(suggestion.estimated_duration_ms)}`
      : "-";
    const methodOptions = methodChoices(suggestion && suggestion.method);
    const isInstruction = Boolean(
      suggestion && (suggestion.action === "cut_segment" || suggestion.action === "manual")
    );
    const primaryBlock = !suggestion
      ? ""
      : isInstruction
        ? `<div class="edit-instruction">${escapeHtml(suggestion.edit_instruction || suggestion.after_text || "")}</div>`
        : `<textarea data-role="after-text" data-suggestion-id="${suggestion.id}">${escapeHtml(suggestion.after_text)}</textarea>`;
    const editNote = suggestion && !isInstruction && suggestion.edit_instruction
      ? `<p class="edit-note">剪辑：${escapeHtml(suggestion.edit_instruction)}</p>`
      : "";
    const sellingPoint = suggestion && suggestion.kept_selling_point
      ? `<span class="selling-point">保留卖点：${escapeHtml(suggestion.kept_selling_point)}</span>`
      : "";
    const handling = state.planMap ? state.planMap[finding.id] : null;
    let handlingBlock = "";
    if (handling && handling.route === "wan") {
      handlingBlock = `
        <div class="handling wan">
          <span class="badge">可自动修改</span>
          <span class="muted">${escapeHtml(handling.reason || "整镜 Wan 重生成")}</span>
          <button class="btn small primary" type="button" data-action="regenerate" data-finding-id="${finding.id}">Wan 重生成（共 ${state.planWanCount || 1} 处）</button>
        </div>`;
    } else if (suggestion) {
      const manualLabel = suggestion.action === "cut_segment" || suggestion.method === "CUT_BROLL"
        ? "删句 / 换镜头（人工）"
        : "转人工";
      handlingBlock = `
        <div class="handling manual">
          <span class="badge">${manualLabel}</span>
          <span class="muted">${escapeHtml((handling && handling.reason) || "工作台不自动处理，请在剪映完成")}</span>
        </div>`;
    }
    const alternatives = suggestion && !isInstruction && suggestion.alternatives && suggestion.alternatives.length
      ? `<div class="alternatives"><button class="btn small ghost" type="button" data-action="restore" data-suggestion-id="${suggestion.id}" data-text="${escapeHtml(suggestion.after_text)}">主文案</button>${suggestion.alternatives.map((text, index) => `<button class="btn small ghost" type="button" data-action="alternative" data-suggestion-id="${suggestion.id}" data-text="${escapeHtml(text)}">备选 ${index + 1}</button>`).join("")}</div>`
      : "";

    card.innerHTML = `
      <div class="finding-head">
        <div class="finding-title">
          <span class="rule">${escapeHtml(finding.rule_id)}</span>
          <strong>${escapeHtml(finding.title)}</strong>
          <span class="badge ${finding.severity}">${severityLabel(finding.severity)}</span>
          ${finding.source === "semantic" ? '<span class="badge semantic">语义事件</span>' : ""}
          ${finding.source === "vision" ? '<span class="badge vision">视觉事件</span>' : ""}
          ${finding.needs_manual_time ? '<span class="badge info">需人工定位</span>' : ""}
        </div>
        <button class="link-btn timecode" type="button" data-action="seek" data-seek-ms="${finding.start_ms}">${fmtTime(finding.start_ms)} - ${fmtTime(finding.end_ms)}</button>
      </div>
      ${["semantic", "vision"].includes(finding.source) && finding.evidence && finding.evidence.length ? `
        <div class="evidence-list">
          ${finding.evidence.map((item) => `
            <div class="evidence-item">
              <span class="tc">${fmtTime(item.start_ms)}</span>
              <span>${escapeHtml(item.quote)}</span>
            </div>
          `).join("")}
        </div>
      ` : `<blockquote>${escapeHtml(finding.quote)}</blockquote>`}
      <p class="rationale">${escapeHtml(finding.rationale)}</p>
      ${suggestion ? `
        <div class="suggestion">
          <div class="suggestion-head">
            <span>${isInstruction ? "剪辑/处理方案" : "口播文案"}</span>
            <span class="abc">${escapeHtml(classLabel(suggestion.abc_class))}</span>
            <span class="method">${escapeHtml(handling && handling.route !== "wan" ? "转人工" : methodLabel(suggestion.method))}</span>
            <span class="duration ${suggestion.duration_warning ? "warn" : ""}">${durationText}${suggestion.duration_warning ? " 超时长" : ""}</span>
            <span>置信度 ${Math.round((suggestion.confidence || 0) * 100)}%</span>
            ${suggestion.status === "accepted" ? '<span class="accepted">已采用</span>' : ""}
            ${suggestion.status === "false_positive" ? '<span class="false-positive">已标记误报</span>' : ""}
          </div>
          ${primaryBlock}
          ${editNote}
          ${sellingPoint}
          ${alternatives}
          ${handlingBlock}
          <div class="suggestion-actions">
            <select data-role="method" data-suggestion-id="${suggestion.id}">${methodOptions}</select>
            <select data-role="constraint" data-suggestion-id="${suggestion.id}">
              <option value="">刷新方式</option>
              <option value="more_conservative">更保守</option>
              <option value="more_colloquial">更口语</option>
              <option value="change_selling_point">换卖点</option>
              <option value="shorter">更短</option>
            </select>
            <button class="btn small" type="button" data-action="refresh" data-suggestion-id="${suggestion.id}">刷新建议</button>
            <button class="btn small" type="button" data-action="save" data-suggestion-id="${suggestion.id}">保存</button>
            ${suggestion.status === "false_positive" ? "" : `<button class="btn small ghost" type="button" data-action="false-positive" data-suggestion-id="${suggestion.id}">误报</button>`}
            <button class="btn small primary" type="button" data-action="accept" data-suggestion-id="${suggestion.id}">采用</button>
          </div>
        </div>
      ` : ""}
    `;
    container.appendChild(card);
  }
}

async function saveSuggestion(suggestionId, accept = false) {
  const textarea = document.querySelector(`textarea[data-suggestion-id="${suggestionId}"]`);
  const method = document.querySelector(`select[data-role="method"][data-suggestion-id="${suggestionId}"]`);
  const payload = {
    after_text: textarea ? textarea.value : undefined,
    method: method ? method.value : undefined,
    status: accept ? "accepted" : undefined,
  };
  const result = await api(`/api/suggestions/${encodeURIComponent(suggestionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.report = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}`);
  renderReport();
  toast(accept ? "已采用这条建议" : "建议已保存");
}

async function refreshSuggestion(suggestionId) {
  const constraint = document.querySelector(`select[data-role="constraint"][data-suggestion-id="${suggestionId}"]`);
  const result = await api(`/api/suggestions/${encodeURIComponent(suggestionId)}/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ constraint: constraint ? constraint.value : "" }),
  });
  state.report = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}`);
  renderReport();
  toast("建议已刷新");
}

async function markFalsePositive(suggestionId) {
  const result = await api(`/api/suggestions/${encodeURIComponent(suggestionId)}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "false_positive", reason: "用户在工作台标记误报" }),
  });
  if (result.report) {
    state.report = result.report;
    renderReport();
  }
  toast("已标记误报，会用于规则校准");
}

async function reviewAgain() {
  if (!state.selectedMaterialId) return;
  const result = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/review?background=true`, {
    method: "POST",
  });
  startJobPolling(result.job_id, { materialId: state.selectedMaterialId });
  toast("审核任务已提交");
}

async function transcribeMaterial() {
  if (!state.selectedMaterialId) return;
  const button = $("#transcribeBtn");
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "识别中…";
  try {
    const report = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/transcribe`, {
      method: "POST",
    });
    state.report = report;
    renderReport();
    await loadMaterials();
    if (report.material.status === "asr_failed") {
      toast(`字幕识别失败：${report.material.asr_error || "未识别到人声"}`, true);
      return;
    }
    const job = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/review?background=true`, {
      method: "POST",
    });
    startJobPolling(job.job_id, { materialId: state.selectedMaterialId });
    toast("字幕识别完成，已提交审核");
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

async function reviewBatch() {
  const batchId = $("#batchSelect").value;
  if (!batchId) {
    toast("请先选择批次", true);
    return;
  }
  const result = await api(`/api/batches/${encodeURIComponent(batchId)}/review?background=true`, {
    method: "POST",
  });
  startJobPolling(result.job_id, { batchId });
  toast("批量审核任务已提交");
}

async function confirmMaterial() {
  if (!state.selectedMaterialId) return;
  state.report = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ accepted: true }),
  });
  renderReport();
  await loadMaterials();
  toast("已确认审核");
}

function exportReport() {
  if (!state.selectedMaterialId) return;
  window.open(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/report.json`, "_blank");
}

async function dryRun() {
  if (!state.selectedMaterialId) return;
  const plan = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/apply?dry_run=true`, { method: "POST" });
  const panel = $("#planPanel");
  panel.classList.remove("hidden");
  const items = (plan.plan || []).map((item) => {
    const label = {
      replace_script: "Wan 口播改写",
      cut: "删段/换镜头",
      manual: "人工处理",
    }[item.op] || item.op;
    const detail = item.text || item.instruction || item.reason || "";
    const range = item.start_ms !== undefined
      ? `${fmtTime(item.start_ms)} - ${fmtTime(item.end_ms)} `
      : "";
    return `<div class="plan-item"><code>${escapeHtml(label)}</code> ${escapeHtml(range)}${escapeHtml(detail)}</div>`;
  }).join("");
  const warnings = (plan.warnings || []).map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
  panel.innerHTML = `
    <h3>修改方案预览（dry-run）</h3>
    <div class="plan-meta">
      <span>预计修改时长：${fmtDuration(plan.estimated_duration_ms)}</span>
      <span>预计成本：¥${Number(plan.estimated_cost || 0).toFixed(2)}</span>
      <span>${escapeHtml(plan.cost_note || "")}</span>
    </div>
    <div class="plan-list">${items || '<p class="muted">没有可执行的修改项。</p>'}</div>
    ${warnings ? `<ul class="warnings">${warnings}</ul>` : ""}
  `;
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function renderMaterial() {
  if (!state.selectedMaterialId) return;
  const plan = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/apply?dry_run=true`, {
    method: "POST",
  });
  const wanOps = (plan.operations || []).filter((op) => op.op === "replace_script" && op.route === "wan");
  if (!wanOps.length) {
    toast("没有可自动改写的口播，请导出人工清单到剪映处理", true);
    return;
  }
  if (!window.confirm(`将用 Wan 重生成 ${wanOps.length} 个镜头，每个约 3–5 分钟。继续？`)) return;
  const result = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/apply?dry_run=false`, {
    method: "POST",
  });
  startJobPolling(result.job_id, { materialId: state.selectedMaterialId, render: true });
  toast(`已提交 ${wanOps.length} 个镜头，预计 ${wanOps.length * 4} 分钟左右`);
}

function downloadWorklist(kind) {
  if (!state.selectedMaterialId) return;
  window.open(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/worklist.${kind}`, "_blank");
}

function downloadOutput() {
  if (!state.selectedMaterialId) return;
  window.open(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/output`, "_blank");
}

function shotBoundaryWarning(text, original) {
  const clean = (value) => String(value || "").replace(/[\s，。！？；：、,.!?;:]/g, "");
  const a = clean(text);
  const b = clean(original);
  if (a.length < 5 || b.length < 5) return "";
  if (a.slice(0, 5) === b.slice(0, 5) || a.slice(-5) === b.slice(-5)) {
    return "新文案首尾与原句相同，Wan 容易顺着原句复读，请换开头/结尾";
  }
  return "";
}

function seekTo(ms) {
  const player = $("#videoPlayer");
  player.currentTime = Math.max(0, (Number(ms) || 0) / 1000 + 0.05);
  player.play().catch(() => {});
}

async function regenerateFinding(findingId) {
  if (!state.selectedMaterialId) return;
  const entry = (state.planMap || {})[findingId];
  if (!entry || entry.route !== "wan") {
    toast("该问题需要人工处理", true);
    return;
  }
  const finding = (state.report?.findings || []).find((item) => item.id === findingId);
  const suggestion = finding && finding.suggestion;
  const textarea = document.querySelector(`textarea[data-suggestion-id="${suggestion ? suggestion.id : ""}"]`);
  const text = ((textarea && textarea.value) || (suggestion && suggestion.after_text) || "").trim();
  if (!text) {
    toast("先填写替换后的台词", true);
    return;
  }
  if (suggestion && text !== (suggestion.after_text || "").trim()) {
    await api(`/api/suggestions/${encodeURIComponent(suggestion.id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ after_text: text }),
    });
    state.report = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}`);
    state.planMap = await loadPlanMap(state.selectedMaterialId);
    renderReport();
  }
  const wanCount = state.planWanCount || 1;
  if (!window.confirm(`将一次性重生成 ${wanCount} 个可自动改写镜头（每个约 3–5 分钟），其余问题保持原样。继续？`)) {
    return;
  }
  const result = await api(`/api/materials/${encodeURIComponent(state.selectedMaterialId)}/apply?dry_run=false`, {
    method: "POST",
  });
  startJobPolling(result.job_id, { materialId: state.selectedMaterialId, render: true });
  toast(`已提交 ${wanCount} 个镜头，预计 ${wanCount * 4} 分钟左右`);
}

function bindEvents() {
  $("#newBatchBtn").addEventListener("click", () => $("#batchForm").classList.toggle("hidden"));
  $("#batchForm").addEventListener("submit", createBatch);
  $("#uploadForm").addEventListener("submit", (event) => uploadMaterial(event).catch((error) => toast(error.message, true)));
  $("#refreshBtn").addEventListener("click", () => loadBatches().catch((error) => toast(error.message, true)));
  $("#batchSelect").addEventListener("change", () => loadMaterials().catch((error) => toast(error.message, true)));
  $("#reviewBtn").addEventListener("click", () => reviewAgain().catch((error) => toast(error.message, true)));
  $("#transcribeBtn").addEventListener("click", () => transcribeMaterial().catch((error) => toast(error.message, true)));
  $("#batchReviewBtn").addEventListener("click", () => reviewBatch().catch((error) => toast(error.message, true)));
  $("#confirmBtn").addEventListener("click", () => confirmMaterial().catch((error) => toast(error.message, true)));
  $("#dryRunBtn").addEventListener("click", () => dryRun().catch((error) => toast(error.message, true)));
  $("#renderBtn").addEventListener("click", () => renderMaterial().catch((error) => toast(error.message, true)));
  $("#downloadBtn").addEventListener("click", downloadOutput);
  $("#exportBtn").addEventListener("click", exportReport);
  $("#worklistMdBtn").addEventListener("click", () => downloadWorklist("md"));
  $("#worklistSrtBtn").addEventListener("click", () => downloadWorklist("srt"));
  $("#videoPlayer").addEventListener("timeupdate", syncTranscriptToVideo);
  $("#videoPlayer").addEventListener("seeked", syncTranscriptToVideo);

  $("#transcriptBody").addEventListener("click", (event) => {
    const row = event.target.closest(".transcript-line[data-start]");
    if (row) seekTranscript(row);
  });

  $("#findings").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const suggestionId = button.dataset.suggestionId;
    if (button.dataset.action === "save") {
      saveSuggestion(suggestionId).catch((error) => toast(error.message, true));
    } else if (button.dataset.action === "accept") {
      saveSuggestion(suggestionId, true).catch((error) => toast(error.message, true));
    } else if (button.dataset.action === "refresh") {
      refreshSuggestion(suggestionId).catch((error) => toast(error.message, true));
    } else if (button.dataset.action === "false-positive") {
      markFalsePositive(suggestionId).catch((error) => toast(error.message, true));
    } else if (button.dataset.action === "seek") {
      seekTo(button.dataset.seekMs);
    } else if (button.dataset.action === "regenerate") {
      regenerateFinding(button.dataset.findingId).catch((error) => toast(error.message, true));
    } else if (button.dataset.action === "alternative" || button.dataset.action === "restore") {
      const textarea = document.querySelector(`textarea[data-suggestion-id="${suggestionId}"]`);
      if (textarea) {
        textarea.value = button.dataset.text || "";
        textarea.focus();
      }
    }
  });


}

bindEvents();
loadBatches().catch((error) => toast(error.message, true));
