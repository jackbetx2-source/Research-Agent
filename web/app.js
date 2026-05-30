const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const form = $("#researchForm");
const topicInput = $("#topicInput");
const runButton = $("#runButton");
const researchFileInput = $("#researchFileInput");
const researchFileSummary = $("#researchFileSummary");
const clearResearchFilesButton = $("#clearResearchFilesButton");
const copyButton = $("#copyButton");
const downloadButton = $("#downloadButton");
const downloadFormat = createDownloadFormatControl();
const statusText = $("#statusText");
const resultOutput = $("#resultOutput");
const referenceList = $("#referenceList");
const analysisStatus = $("#analysisStatus");
const analysisTableBody = $("#analysisTableBody");
const analysisSummary = $("#analysisSummary");
const analysisPrompt = $("#analysisPrompt");
const startAnalysisButton = $("#startAnalysisButton");
const literatureForm = $("#literatureForm");
const doiInput = $("#doiInput");
const doiAnalyzeButton = $("#doiAnalyzeButton");
const pdfInput = $("#pdfInput");
const pdfFileSummary = $("#pdfFileSummary");
const clearPdfButton = $("#clearPdfButton");
const literatureStatus = $("#literatureStatus");
const doiAnalysisBody = $("#doiAnalysisBody");
const doiSummary = $("#doiSummary");
const slrForm = $("#slrForm");
const slrTopicInput = $("#slrTopicInput");
const slrCountInput = $("#slrCountInput");
const slrCategoryInput = $("#slrCategoryInput");
const slrStartDateInput = $("#slrStartDateInput");
const slrEndDateInput = $("#slrEndDateInput");
const slrCitationInput = $("#slrCitationInput");
const slrUseSearchInput = $("#slrUseSearchInput");
const slrRunButton = $("#slrRunButton");
const slrStatus = $("#slrStatus");
const slrSummary = $("#slrSummary");
const slrOutput = $("#slrOutput");
const slrUploadPanel = $("#slrUploadPanel");
const slrPdfInput = $("#slrPdfInput");
const slrPdfSummary = $("#slrPdfSummary");
const clearSlrPdfButton = $("#clearSlrPdfButton");
const slrReferenceInput = $("#slrReferenceInput");
const chatForm = $("#chatForm");
const chatInput = $("#chatInput");
const chatMessages = $("#chatMessages");
const chatStatus = $("#chatStatus");
const chatSendButton = $("#chatSendButton");

let latestMarkdown = "";
let latestTopic = "research-report";
let latestReferences = [];
let latestAnalysisRows = [];
let latestAnalysisSummary = null;
let latestSlrMarkdown = "";
let selectedResearchFiles = [];
let selectedPdfFiles = [];
let selectedSlrPdfFiles = [];
let chatHistory = [];
const appBasePath = (window.location.pathname.match(/^\/v\d+(?=\/|$)/) || [""])[0];

initPageNavigation();
initSplitResizers();
initSlrDatePresets();
initSlrSourceModes();
topicInput.required = false;

function createDownloadFormatControl() {
  const select = document.createElement("select");
  select.id = "downloadFormat";
  select.setAttribute("aria-label", "下载格式");
  select.disabled = true;
  select.innerHTML = '<option value="txt" selected>TXT</option><option value="pdf">PDF</option>';

  const label = document.createElement("label");
  label.className = "download-format";
  label.appendChild(select);
  downloadButton.textContent = "下载";
  downloadButton.parentNode.insertBefore(label, downloadButton.nextSibling);
  return select;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const topic = topicInput.value.trim();
  if (!topic && !selectedResearchFiles.length) {
    setStatus("请输入研究主题，或上传 PDF / DOCX 文件。", true);
    return;
  }

  setResearchRunning(true);
  setResultActionsEnabled(false);
  setStatus("研究正在运行，通常需要 20-90 秒...");
  renderMarkdown(resultOutput, "## 正在生成\n\n请稍候，最终报告会显示在这里。");
  renderReferences([]);
  latestAnalysisRows = [];
  latestAnalysisSummary = null;
  updateAnalysisPrompt();

  try {
    const response = selectedResearchFiles.length
      ? await submitResearchWithFiles(topic)
      : await fetch(apiPath("/api/research"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, fast: true }),
      });
    let payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (payload.job_id) payload = await waitForJob("/api/research", payload.job_id, setStatus);

    latestMarkdown = payload.final_report || "";
    latestTopic = topic;
    latestReferences = Array.isArray(payload.references) ? payload.references : [];
    renderMarkdown(resultOutput, latestMarkdown || "## 暂无结果");
    renderReferences(latestReferences);
    setStatus(`完成，已保存到 ${payload.output_path || "outputs"}`);
    setResultActionsEnabled(Boolean(latestMarkdown));
    updateAnalysisPrompt();
    if (selectedResearchFiles.length) clearSelectedResearchFiles();
  } catch (error) {
    latestMarkdown = "";
    latestReferences = [];
    renderMarkdown(resultOutput, `## 运行失败\n\n${error.message}`);
    renderReferences([]);
    setStatus("运行失败。请检查本地服务、网络或模型配置。", true);
    updateAnalysisPrompt();
  } finally {
    setResearchRunning(false);
  }
});

researchFileInput.addEventListener("change", () => {
  const allFiles = Array.from(researchFileInput.files || []);
  const legacyWordFiles = allFiles.filter((file) => /\.doc$/i.test(file.name));
  if (legacyWordFiles.length) {
    setStatus(`不支持旧版 .doc 文件：${legacyWordFiles.map((file) => file.name).join("、")}。请另存为 .docx 或 PDF 后再上传。`, true);
  }
  const unsupportedFiles = allFiles.filter((file) => !isResearchDocument(file) && !/\.doc$/i.test(file.name));
  if (unsupportedFiles.length) {
    setStatus(`不支持这些文件：${unsupportedFiles.map((file) => file.name).join("、")}。目前只支持 PDF / DOCX。`, true);
  }
  const files = allFiles.filter(isResearchDocument);
  addSelectedResearchFiles(files);
  researchFileInput.value = "";
});

clearResearchFilesButton.addEventListener("click", () => {
  clearSelectedResearchFiles();
});

slrForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const topic = slrTopicInput.value.trim();
  if (!topic) return;

  setSlrRunning(true);
  slrStatus.textContent = "正在启动文献综述...";
  renderLiteratureSummary(slrSummary, null);
  renderMarkdown(slrOutput, "## 正在生成文献综述\n\n系统正在准备文献来源并进行综合分析。");

  try {
    const hasUserSources = selectedSlrPdfFiles.length || parseLiteratureLinkInput(slrReferenceInput.value).length;
    const slrUserContext = extractLiteratureFreeText(slrReferenceInput.value);
    const effectiveTopic = slrUserContext ? `${topic}\n\nUser additional instructions/context:\n${slrUserContext}` : topic;
    const effectiveSourceMode = currentSlrSourceMode();
    const response = effectiveSourceMode === "search"
      ? await submitSearchOnlySlr(effectiveTopic, effectiveSourceMode)
      : await submitUploadSlr(topic, effectiveSourceMode);
    let payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (payload.job_id) {
      payload = await waitForJob("/api/slr", payload.job_id, (message) => {
        slrStatus.textContent = message;
      });
    }

    latestSlrMarkdown = payload.report_markdown || "";
    renderSlrSummary(payload);
    renderMarkdown(slrOutput, latestSlrMarkdown || "## 暂无文献综述结果");
    slrStatus.textContent = `已完成，共综述 ${(payload.papers || []).length} 篇/项文献。`;
    if (hasUserSources) clearSelectedSlrPdfFiles();
  } catch (error) {
    latestSlrMarkdown = "";
    renderLiteratureSummary(slrSummary, null);
    renderMarkdown(slrOutput, `## 文献综述失败\n\n${error.message}`);
    slrStatus.textContent = "文献综述失败";
  } finally {
    setSlrRunning(false);
  }
});

slrPdfInput.addEventListener("change", () => {
  const files = filterSupportedUploadFiles(Array.from(slrPdfInput.files || []), (message) => {
    slrStatus.textContent = message;
  });
  addSelectedSlrPdfFiles(files);
  slrPdfInput.value = "";
});

clearSlrPdfButton.addEventListener("click", () => {
  clearSelectedSlrPdfFiles();
});

startAnalysisButton.addEventListener("click", async () => {
  if (!latestReferences.length || !latestMarkdown) {
    updateAnalysisPrompt();
    return;
  }
  setAnalysisRunning(true);
  renderAnalysisRows([]);
  renderLiteratureSummary(analysisSummary, null);
  analysisStatus.textContent = "正在启动文献分析...";

  try {
    const response = await fetch(apiPath("/api/literature-analysis"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic: latestTopic,
        references: latestReferences,
        final_report: latestMarkdown,
      }),
    });
    let payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (payload.job_id) payload = await waitForJob("/api/literature-analysis", payload.job_id, (message) => {
      analysisStatus.textContent = message;
    });

    latestAnalysisRows = Array.isArray(payload.rows) ? payload.rows : [];
    latestAnalysisSummary = normalizeLiteratureSummary(payload.summary);
    renderAnalysisRows(latestAnalysisRows);
    renderLiteratureSummary(analysisSummary, latestAnalysisSummary);
    analysisStatus.textContent = `已分析 ${latestAnalysisRows.length} 篇/项文献`;
    updateAnalysisPrompt();
  } catch (error) {
    latestAnalysisRows = [];
    latestAnalysisSummary = null;
    renderAnalysisRows([]);
    renderLiteratureSummary(analysisSummary, null);
    analysisStatus.textContent = "文献分析失败";
    analysisPrompt.textContent = `文献分析失败：${error.message}`;
  } finally {
    setAnalysisRunning(false);
  }
});

literatureForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const entries = parseLiteratureLinkInput(doiInput.value);
  const userContext = extractLiteratureFreeText(doiInput.value);
  const pdfFiles = selectedPdfFiles;
  if (!entries.length && !pdfFiles.length && !userContext) {
    literatureStatus.textContent = "请先提供 DOI、链接、文字说明或 PDF / DOCX";
    return;
  }

  const linkReferences = entries.map(literatureLinkToReference);
  const previewReferences = [...linkReferences, ...pdfFiles.map(pdfToReference)];
  renderDoiPendingRows(previewReferences);
  renderLiteratureSummary(doiSummary, null);
  setDoiAnalysisRunning(true);
  literatureStatus.textContent = "正在启动综合文献分析...";

  try {
    const response = pdfFiles.length
      ? await submitCombinedLiteratureAnalysis(linkReferences, pdfFiles, userContext)
      : await submitLinkLiteratureAnalysis(linkReferences, userContext);
    let payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (payload.job_id) payload = await waitForJob("/api/literature-analysis", payload.job_id, (message) => {
      literatureStatus.textContent = message;
    });

    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    renderDoiAnalysisRows(rows, previewReferences);
    renderLiteratureSummary(doiSummary, normalizeLiteratureSummary(payload.summary));
    literatureStatus.textContent = `已分析 ${rows.length} 篇/项文献`;
    clearSelectedPdfFiles();
  } catch (error) {
    renderDoiErrorRows(previewReferences, error.message);
    renderLiteratureSummary(doiSummary, null);
    literatureStatus.textContent = "综合文献分析失败";
  } finally {
    setDoiAnalysisRunning(false);
  }
});

pdfInput.addEventListener("change", () => {
  const files = filterSupportedUploadFiles(Array.from(pdfInput.files || []), (message) => {
    literatureStatus.textContent = message;
  });
  addSelectedPdfFiles(files);
  pdfInput.value = "";
});

clearPdfButton.addEventListener("click", () => {
  clearSelectedPdfFiles();
});

copyButton.addEventListener("click", async () => {
  if (!latestMarkdown) return;
  await navigator.clipboard.writeText(latestMarkdown);
  setStatus("结果已复制。");
});

downloadButton.addEventListener("click", () => {
  if (!latestMarkdown) return;
  if (downloadFormat.value === "pdf") {
    downloadPdfReport();
    return;
  }
  downloadText(`${safeFileName(latestTopic)}.txt`, buildDownloadText());
  setStatus("TXT 报告已下载。");
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  appendChatMessage("user", message);
  chatHistory.push({ role: "user", content: message });
  chatInput.value = "";
  setChatRunning(true);
  chatStatus.textContent = "LLM 正在回答...";

  try {
    const response = await fetch(apiPath("/api/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: chatHistory.slice(-8) }),
    });
    const payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    const answer = payload.answer || "未返回有效回答。";
    appendChatMessage("assistant", answer);
    chatHistory.push({ role: "assistant", content: answer });
    chatStatus.textContent = "已回答。";
  } catch (error) {
    appendChatMessage("assistant", `回答失败：${error.message}`);
    chatStatus.textContent = "回答失败";
  } finally {
    setChatRunning(false);
  }
});

function initPageNavigation() {
  const initialPage = pageFromHash();
  showPage(initialPage, false);
  $$("[data-page-target]").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.pageTarget));
  });
  window.addEventListener("hashchange", () => showPage(pageFromHash(), false));
}

function showPage(pageName, updateHash = true) {
  const allowed = ["workspace", "slr", "literature", "analysis", "about"];
  const nextPage = allowed.includes(pageName) ? pageName : "workspace";
  $$("[data-page]").forEach((view) => view.classList.toggle("is-active", view.dataset.page === nextPage));
  $$("[data-page-target]").forEach((button) => {
    const active = button.dataset.pageTarget === nextPage;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  });
  if (updateHash && window.location.hash !== `#${nextPage}`) {
    history.pushState(null, "", `#${nextPage}`);
  }
  if (nextPage === "analysis") updateAnalysisPrompt();
}

function pageFromHash() {
  const value = window.location.hash.replace("#", "");
  return value || "workspace";
}

function setResearchRunning(isRunning) {
  runButton.disabled = isRunning;
  topicInput.disabled = isRunning;
  researchFileInput.disabled = isRunning;
  clearResearchFilesButton.disabled = isRunning || !selectedResearchFiles.length;
  runButton.textContent = isRunning ? "研究中..." : "开始研究";
}

function setSlrRunning(isRunning) {
  slrRunButton.disabled = isRunning;
  slrTopicInput.disabled = isRunning;
  [slrCountInput, slrCategoryInput, slrStartDateInput, slrEndDateInput, slrCitationInput, slrUseSearchInput].forEach((field) => {
    field.disabled = isRunning;
  });
  slrRunButton.textContent = isRunning ? "生成中..." : "生成文献综述";
}

function setResultActionsEnabled(isEnabled) {
  copyButton.disabled = !isEnabled;
  downloadButton.disabled = !isEnabled;
  downloadFormat.disabled = !isEnabled;
}

function setAnalysisRunning(isRunning) {
  startAnalysisButton.disabled = isRunning || !latestReferences.length || !latestMarkdown;
  startAnalysisButton.textContent = isRunning ? "分析中..." : "开始文献分析";
}

function setDoiAnalysisRunning(isRunning) {
  doiInput.disabled = isRunning;
  doiAnalyzeButton.disabled = isRunning;
  pdfInput.disabled = isRunning;
  clearPdfButton.disabled = isRunning || !selectedPdfFiles.length;
  doiAnalyzeButton.textContent = isRunning ? "分析中..." : "开始综合分析";
}

function setChatRunning(isRunning) {
  chatInput.disabled = isRunning;
  chatSendButton.disabled = isRunning;
  chatSendButton.textContent = isRunning ? "发送中..." : "发送";
}

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

function renderReferences(references) {
  if (!references.length) {
    referenceList.innerHTML = '<li class="empty-state">暂无参考文献。</li>';
    return;
  }
  referenceList.innerHTML = references.map((reference) => `
    <li class="reference-item">
      <strong>${linkOrText(reference.title || "未命名来源", reference.source)}</strong>
      <span>${escapeHtml(reference.branch_name || "研究分支")}</span>
      <p>${escapeHtml(reference.relevance || "未说明")}</p>
    </li>
  `).join("");
}

function renderAnalysisRows(rows) {
  if (!rows.length) {
    analysisTableBody.innerHTML = '<tr><td colspan="8" class="empty-state">确认后会启动 LLM 工作流生成文献分析表。</td></tr>';
    return;
  }
  analysisTableBody.innerHTML = rows.map((row) => `
    <tr>
      <th>${linkOrText(row.title || "未命名文献", row.source)}</th>
      <td>${reviewCell(row.contribution || row.innovation)}</td>
      <td>${reviewCell(row.methodology || row.method)}</td>
      <td>${reviewCell(row.evidence_strength)}</td>
      <td>${reviewCell(row.strengths)}</td>
      <td>${reviewCell(row.limitations || row.weaknesses || row.limitation)}</td>
      <td>${reviewCell(row.literature_positioning)}</td>
      <td>${reviewCellWithMeta(row.actionable_suggestions || row.next_step, row.confidence)}</td>
    </tr>
  `).join("");
}

function renderDoiPendingRows(references) {
  doiAnalysisBody.innerHTML = references.map((reference) => `
    <tr>
      <th>${linkOrText(reference.title || "待分析文献", reference.source)}</th>
      <td>等待 LLM 分析</td><td>等待 LLM 分析</td><td>等待 LLM 分析</td><td>等待 LLM 分析</td>
      <td>等待 LLM 分析</td><td>等待 LLM 分析</td><td>排队中</td>
    </tr>
  `).join("");
}

function renderDoiAnalysisRows(rows, fallbackReferences) {
  if (!rows.length) {
    doiAnalysisBody.innerHTML = '<tr><td colspan="8" class="empty-state">文献分析已完成，但没有返回可展示结果。</td></tr>';
    return;
  }
  doiAnalysisBody.innerHTML = rows.map((row, index) => {
    const fallback = fallbackReferences[index] || {};
    return `
      <tr>
        <th>${linkOrText(row.title || fallback.title || "未命名文献", row.source || fallback.source)}</th>
        <td>${reviewCell(row.contribution || row.innovation)}</td>
        <td>${reviewCell(row.methodology || row.method)}</td>
        <td>${reviewCell(row.evidence_strength)}</td>
        <td>${reviewCell(row.strengths)}</td>
        <td>${reviewCell(row.limitations || row.weaknesses || row.limitation)}</td>
        <td>${reviewCell(row.literature_positioning)}</td>
        <td>${reviewCellWithMeta(row.actionable_suggestions || row.next_step || "已完成", row.confidence)}</td>
      </tr>
    `;
  }).join("");
}

function renderDoiErrorRows(references, message) {
  doiAnalysisBody.innerHTML = references.map((reference) => `
    <tr>
      <th>${linkOrText(reference.title, reference.source)}</th>
      <td>分析失败</td><td>未生成</td><td>未生成</td><td>未生成</td><td>未生成</td>
      <td>${escapeHtml(message)}</td><td>失败</td>
    </tr>
  `).join("");
}

function renderLiteratureSummary(container, summary) {
  if (!container) return;
  const normalized = normalizeLiteratureSummary(summary);
  if (!normalized) {
    container.classList.add("is-empty");
    container.classList.remove("is-resizable");
    updateSplitResizerVisibility(container);
    container.innerHTML = "";
    return;
  }
  const groups = [
    ["共同优势", normalized.common_strengths],
    ["共同弱点", normalized.common_weaknesses],
    ["方法模式", normalized.methodological_patterns],
    ["证据缺口", normalized.evidence_gaps],
    ["研究空白", normalized.research_gaps],
    ["来源与引用", normalized.recommended_reading_order],
    ["后续行动", normalized.next_actions],
  ].filter(([, items]) => items.length);
  container.classList.remove("is-empty");
  container.classList.add("is-resizable");
  container.innerHTML = `
    <h3>跨文献总结</h3>
    <p class="summary-lead">${escapeHtml(normalized.overall_assessment || "已生成跨文献总结。")}</p>
    <div class="summary-grid">
      ${groups.map(([title, items]) => summaryGroup(title, items)).join("")}
      ${normalized.confidence ? summaryGroup("置信度", [normalized.confidence]) : ""}
    </div>
  `;
  updateSplitResizerVisibility(container);
}

function renderSlrSummary(payload) {
  const synthesis = payload.synthesis || {};
  const themes = Array.isArray(synthesis.themes) ? synthesis.themes.map((theme) => `${theme.name}: ${theme.summary}`) : [];
  const sourceCounts = payload.source_counts && typeof payload.source_counts === "object"
    ? Object.entries(payload.source_counts).map(([source, count]) => `${source}: ${count}`)
    : [];
  const sourceErrors = payload.source_errors && typeof payload.source_errors === "object"
    ? Object.entries(payload.source_errors).map(([source, reason]) => sourceStatusMessage(source, reason))
    : [];
  renderLiteratureSummary(slrSummary, {
    overall_assessment: synthesis.executive_summary || `已生成 ${(payload.papers || []).length} 篇论文的文献综述。`,
    methodological_patterns: synthesis.methodological_patterns || [],
    research_gaps: synthesis.gaps || [],
    common_strengths: themes,
    common_weaknesses: synthesis.disagreements || [],
    evidence_gaps: synthesis.convergences || [],
    recommended_reading_order: sourceCounts,
    next_actions: [`完整报告已保存到 ${payload.output_path || "outputs"}`, ...sourceErrors],
    confidence: `检索词：${payload.search_query || ""}；引用格式：${payload.citation_format || ""}；来源：arXiv / OpenAlex / PubMed / Crossref`,
  });
}

function initSplitResizers() {
  document.querySelectorAll(".split-resizer[data-resize-target]").forEach((handle) => {
    const target = document.getElementById(handle.dataset.resizeTarget);
    if (!target) return;
    handle.addEventListener("pointerdown", (event) => startSplitResize(event, handle, target));
    handle.addEventListener("keydown", (event) => {
      if (!["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const current = Number.parseInt(target.style.getPropertyValue("--summary-height"), 10) ||
        Math.round(target.getBoundingClientRect().height);
      let next = current;
      if (event.key === "ArrowUp") next -= 32;
      if (event.key === "ArrowDown") next += 32;
      if (event.key === "Home") next = 140;
      if (event.key === "End") next = 640;
      setSummaryHeight(target, next);
    });
    updateSplitResizerVisibility(target);
  });
}

function initSlrDatePresets() {
  document.querySelectorAll("[data-slr-years]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-slr-years]").forEach((item) => item.classList.remove("is-active"));
      button.classList.add("is-active");
      const years = Number(button.dataset.slrYears || 0);
      if (!years) {
        slrStartDateInput.value = "";
        slrEndDateInput.value = "";
        return;
      }
      const today = new Date();
      const start = new Date(today);
      start.setFullYear(today.getFullYear() - years);
      slrStartDateInput.value = toDateString(start);
      slrEndDateInput.value = toDateString(today);
    });
  });
}

function initSlrSourceModes() {
  slrUseSearchInput.addEventListener("change", updateSlrSourceModeUi);
  slrReferenceInput.addEventListener("input", updateSlrSourceModeUi);
  updateSlrSourceModeUi();
}

function currentSlrSourceMode() {
  const hasUserSources = selectedSlrPdfFiles.length || parseLiteratureLinkInput(slrReferenceInput.value).length;
  if (!hasUserSources) return "search";
  return slrUseSearchInput.checked ? "mixed" : "upload";
}

function updateSlrSourceModeUi() {
  const mode = currentSlrSourceMode();
  if (mode === "upload") {
    slrStatus.textContent = "仅使用你上传或粘贴的文献生成文献综述";
  } else if (mode === "mixed") {
    slrStatus.textContent = "先纳入你的文献，再用公开来源补充";
  } else {
    slrStatus.textContent = "填写主题即可生成，也可附加文件、链接或说明。";
  }
}

async function submitResearchWithFiles(topic) {
  const formData = new FormData();
  formData.append("topic", topic);
  formData.append("fast", "true");
  selectedResearchFiles.forEach((file) => formData.append("document", file));
  return fetch(apiPath("/api/research"), { method: "POST", body: formData });
}

async function submitSearchOnlySlr(topic, sourceMode) {
  return fetch(apiPath("/api/slr"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic,
      source_mode: sourceMode,
      max_results: Number(slrCountInput.value || 20),
      category: slrCategoryInput.value.trim(),
      start_date: normalizeDateInput(slrStartDateInput.value, "start"),
      end_date: normalizeDateInput(slrEndDateInput.value, "end"),
      citation_format: slrCitationInput.value,
    }),
  });
}

async function submitUploadSlr(topic, sourceMode) {
  const entries = parseLiteratureLinkInput(slrReferenceInput.value);
  const userContext = extractLiteratureFreeText(slrReferenceInput.value);
  const references = entries.map((entry) => {
    const reference = literatureLinkToReference(entry);
    reference.source_origin = "user_link";
    reference.source_label = "User link";
    return reference;
  });
  const formData = new FormData();
  formData.append("topic", userContext ? `${topic}\n\nUser additional instructions/context:\n${userContext}` : topic);
  formData.append("source_mode", sourceMode);
  formData.append("max_results", String(Number(slrCountInput.value || 20)));
  formData.append("category", sourceMode === "upload" ? "" : slrCategoryInput.value.trim());
  formData.append("start_date", sourceMode === "upload" ? "" : normalizeDateInput(slrStartDateInput.value, "start"));
  formData.append("end_date", sourceMode === "upload" ? "" : normalizeDateInput(slrEndDateInput.value, "end"));
  formData.append("citation_format", slrCitationInput.value);
  formData.append("references", JSON.stringify(references));
  selectedSlrPdfFiles.forEach((file) => formData.append("pdf", file));
  return fetch(apiPath("/api/slr/upload"), { method: "POST", body: formData });
}

function addSelectedSlrPdfFiles(files) {
  if (!files.length) {
    updateSlrPdfSummary();
    updateSlrSourceModeUi();
    return;
  }
  const existing = new Set(selectedSlrPdfFiles.map(pdfFileKey));
  files.forEach((file) => {
    const key = pdfFileKey(file);
    if (!existing.has(key)) {
      selectedSlrPdfFiles.push(file);
      existing.add(key);
    }
  });
  updateSlrPdfSummary();
  updateSlrSourceModeUi();
}

function clearSelectedSlrPdfFiles() {
  selectedSlrPdfFiles = [];
  slrPdfInput.value = "";
  updateSlrPdfSummary();
  updateSlrSourceModeUi();
}

function updateSlrPdfSummary() {
  if (!selectedSlrPdfFiles.length) {
    slrPdfSummary.textContent = "尚未选择文件。";
    clearSlrPdfButton.disabled = true;
    return;
  }
  slrPdfSummary.textContent = formatSelectedFilesSummary(selectedSlrPdfFiles);
  clearSlrPdfButton.disabled = false;
}

function normalizeDateInput(value, boundary) {
  const cleaned = String(value || "").trim();
  if (!cleaned) return "";
  const yearOnly = cleaned.match(/^(\d{4})$/);
  if (yearOnly) return boundary === "end" ? `${yearOnly[1]}-12-31` : `${yearOnly[1]}-01-01`;
  const yearMonth = cleaned.match(/^(\d{4})[-/.](\d{1,2})$/);
  if (yearMonth) {
    const year = Number(yearMonth[1]);
    const month = Number(yearMonth[2]);
    if (month < 1 || month > 12) return "";
    if (boundary === "end") {
      return toDateString(new Date(year, month, 0));
    }
    return `${yearMonth[1]}-${String(month).padStart(2, "0")}-01`;
  }
  const full = cleaned.match(/^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$/);
  if (!full) return cleaned;
  return `${full[1]}-${String(Number(full[2])).padStart(2, "0")}-${String(Number(full[3])).padStart(2, "0")}`;
}

function toDateString(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function startSplitResize(event, handle, target) {
  if (event.button !== 0) return;
  event.preventDefault();
  const startY = event.clientY;
  const startHeight = target.getBoundingClientRect().height;
  document.body.classList.add("is-resizing-split");
  handle.setPointerCapture(event.pointerId);

  const onMove = (moveEvent) => {
    const nextHeight = startHeight + moveEvent.clientY - startY;
    setSummaryHeight(target, nextHeight);
  };
  const onEnd = () => {
    document.body.classList.remove("is-resizing-split");
    handle.removeEventListener("pointermove", onMove);
    handle.removeEventListener("pointerup", onEnd);
    handle.removeEventListener("pointercancel", onEnd);
  };

  handle.addEventListener("pointermove", onMove);
  handle.addEventListener("pointerup", onEnd);
  handle.addEventListener("pointercancel", onEnd);
}

function setSummaryHeight(target, value) {
  const height = Math.max(140, Math.min(720, Math.round(value)));
  target.style.setProperty("--summary-height", `${height}px`);
}

function updateSplitResizerVisibility(target) {
  const handle = document.querySelector(`.split-resizer[data-resize-target="${target.id}"]`);
  if (!handle) return;
  handle.classList.toggle("is-visible", !target.classList.contains("is-empty"));
}

function sourceStatusMessage(source, reason) {
  const detail = String(reason || "");
  if (/429|rate limit|cooldown|timeout|timed out|超时/i.test(detail)) {
    return `${source} 本次限流或超时，已跳过并使用其他来源继续生成`;
  }
  return `${source} 本次检索未成功，已使用其他来源继续生成`;
}

function normalizeLiteratureSummary(summary) {
  if (!summary || typeof summary !== "object") return null;
  const normalized = {
    overall_assessment: String(summary.overall_assessment || "").trim(),
    common_strengths: toStringList(summary.common_strengths),
    common_weaknesses: toStringList(summary.common_weaknesses),
    methodological_patterns: toStringList(summary.methodological_patterns),
    evidence_gaps: toStringList(summary.evidence_gaps),
    research_gaps: toStringList(summary.research_gaps),
    recommended_reading_order: toStringList(summary.recommended_reading_order),
    next_actions: toStringList(summary.next_actions),
    confidence: String(summary.confidence || "").trim(),
  };
  const hasContent = normalized.overall_assessment || normalized.confidence ||
    Object.values(normalized).some((value) => Array.isArray(value) && value.length);
  return hasContent ? normalized : null;
}

function summaryGroup(title, items) {
  return `<div class="summary-group"><strong>${escapeHtml(title)}</strong><ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>`;
}

function toStringList(value) {
  if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

function updateAnalysisPrompt() {
  if (!latestReferences.length || !latestMarkdown) {
    analysisPrompt.textContent = "请先在研究报告完成一次研究，再启动文献分析。";
    startAnalysisButton.disabled = true;
    analysisStatus.textContent = "等待研究结果";
    return;
  }
  if (latestAnalysisRows.length) {
    analysisPrompt.textContent = "文献分析已完成，可以重新分析。";
    startAnalysisButton.disabled = false;
    startAnalysisButton.textContent = "重新分析";
    return;
  }
  analysisPrompt.textContent = `检测到 ${latestReferences.length} 篇/项相关文献。是否启动 LLM 文献分析工作流？`;
  startAnalysisButton.disabled = false;
  startAnalysisButton.textContent = "开始文献分析";
  analysisStatus.textContent = "等待确认";
}

function parseLiteratureLinkInput(value) {
  const doiMatches = value.match(/10\.\d{4,9}\/[^\s,;，；]+/gi) || [];
  const urlMatches = value.match(/https?:\/\/[^\s,;，；]+/gi) || [];
  const seen = new Set();
  const entries = [];
  doiMatches.forEach((doi) => addLiteratureEntry(entries, seen, { type: "doi", value: doi.replace(/[.)\]}]+$/g, "") }));
  urlMatches.forEach((url) => {
    const cleaned = url.replace(/[.)\]}]+$/g, "");
    const doi = extractDoiFromText(cleaned);
    addLiteratureEntry(entries, seen, doi ? { type: "doi", value: doi } : { type: "url", value: cleaned });
  });
  return entries;
}

function addLiteratureEntry(entries, seen, entry) {
  const key = `${entry.type}:${entry.value.toLowerCase()}`;
  if (seen.has(key)) return;
  seen.add(key);
  entries.push(entry);
}

function extractLiteratureFreeText(value) {
  const doiMatches = value.match(/10\.\d{4,9}\/[^\s,;，；]+/gi) || [];
  const urlMatches = value.match(/https?:\/\/[^\s,;，；]+/gi) || [];
  let text = String(value || "");
  [...doiMatches, ...urlMatches].forEach((token) => {
    text = text.replace(token, " ");
  });
  return text
    .split(/\n{2,}/)
    .map((block) => block.replace(/\s+/g, " ").trim())
    .filter((block) => block.length >= 8)
    .join("\n\n");
}

function extractDoiFromText(value) {
  const match = value.match(/10\.\d{4,9}\/[^\s,;，；]+/i);
  return match ? match[0].replace(/[.)\]}]+$/g, "") : "";
}

function literatureLinkToReference(entry) {
  if (entry.type === "doi") {
    return {
      title: `DOI: ${entry.value}`,
      source: `https://doi.org/${entry.value}`,
      relevance: "用户在文献分析中主动提交，需要进行文献分析。",
      branch_name: "文献分析",
    };
  }
  return {
    title: readableTitleFromUrl(entry.value),
    source: entry.value,
    relevance: "用户在文献分析中主动提交论文链接，需要进行文献分析。",
    branch_name: "文献分析",
  };
}

function pdfToReference(file) {
  return {
    title: file.name.replace(/\.(pdf|docx)$/i, ""),
    source: file.name,
    relevance: "用户上传文件，需要进行文献分析。",
    branch_name: "文件上传",
  };
}

function isResearchDocument(file) {
  return /\.pdf$/i.test(file.name) ||
    /\.docx$/i.test(file.name) ||
    file.type === "application/pdf" ||
    file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
}

function filterSupportedUploadFiles(files, report) {
  const legacyWordFiles = files.filter((file) => /\.doc$/i.test(file.name));
  if (legacyWordFiles.length) {
    report(`不支持旧版 .doc 文件：${legacyWordFiles.map((file) => file.name).join("、")}。请另存为 .docx 或 PDF 后再上传。`);
  }
  const supported = files.filter(isResearchDocument);
  const unsupported = files.filter((file) => !isResearchDocument(file) && !/\.doc$/i.test(file.name));
  if (unsupported.length) {
    report(`不支持这些文件：${unsupported.map((file) => file.name).join("、")}。目前只支持 PDF / DOCX。`);
  }
  return supported;
}

function addSelectedResearchFiles(files) {
  if (!files.length) {
    updateResearchFileSummary();
    return;
  }
  const existing = new Set(selectedResearchFiles.map(pdfFileKey));
  files.forEach((file) => {
    const key = pdfFileKey(file);
    if (!existing.has(key)) {
      selectedResearchFiles.push(file);
      existing.add(key);
    }
  });
  updateResearchFileSummary();
}

function clearSelectedResearchFiles() {
  selectedResearchFiles = [];
  researchFileInput.value = "";
  updateResearchFileSummary();
}

function updateResearchFileSummary() {
  if (!selectedResearchFiles.length) {
    researchFileSummary.textContent = "尚未选择文件。";
    clearResearchFilesButton.disabled = true;
    return;
  }
  researchFileSummary.textContent = formatSelectedFilesSummary(selectedResearchFiles);
  clearResearchFilesButton.disabled = false;
}

function addSelectedPdfFiles(files) {
  if (!files.length) {
    updatePdfFileSummary();
    return;
  }
  const existing = new Set(selectedPdfFiles.map(pdfFileKey));
  files.forEach((file) => {
    const key = pdfFileKey(file);
    if (!existing.has(key)) {
      selectedPdfFiles.push(file);
      existing.add(key);
    }
  });
  updatePdfFileSummary();
}

function clearSelectedPdfFiles() {
  selectedPdfFiles = [];
  pdfInput.value = "";
  updatePdfFileSummary();
}

function updatePdfFileSummary() {
  if (!selectedPdfFiles.length) {
    pdfFileSummary.textContent = "尚未选择文件。";
    clearPdfButton.disabled = true;
    return;
  }
  pdfFileSummary.textContent = formatSelectedFilesSummary(selectedPdfFiles);
  clearPdfButton.disabled = false;
}

function pdfFileKey(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

function readableTitleFromUrl(url) {
  try {
    const parsed = new URL(url);
    const part = decodeURIComponent(parsed.pathname.split("/").filter(Boolean).pop() || parsed.hostname);
    return part.replace(/[-_]+/g, " ") || parsed.hostname;
  } catch {
    return url;
  }
}

async function submitLinkLiteratureAnalysis(references, userContext = "") {
  return fetch(apiPath("/api/literature-analysis"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      topic: "user-provided DOI and literature link analysis",
      references,
      final_report: buildLiteratureUserContext(userContext, "The user provided DOI identifiers or literature links directly in the literature assistant."),
    }),
  });
}

async function submitCombinedLiteratureAnalysis(references, pdfFiles, userContext = "") {
  const formData = new FormData();
  formData.append("references", JSON.stringify(references));
  formData.append("user_context", userContext);
  pdfFiles.forEach((file) => formData.append("pdf", file));
  return fetch(apiPath("/api/literature-analysis/pdf"), { method: "POST", body: formData });
}

function buildLiteratureUserContext(userContext, fallback) {
  const context = String(userContext || "").trim();
  if (!context) return fallback;
  return `${fallback}\n\nUser-provided text context or instructions:\n${context}`;
}

async function waitForJob(basePath, jobId, updateStatus) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 10 * 60 * 1000) {
    await sleep(1000);
    const response = await fetch(apiPath(`${basePath}/${jobId}`), { cache: "no-store" });
    const payload = await readJsonResponse(response);
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (payload.status === "done") return payload;
    if (payload.status === "error") throw new Error(payload.error || "任务运行失败。");
    const elapsed = Math.floor((Date.now() - startedAt) / 1000);
    updateStatus(translateJobStage(payload.stage) || `运行中，已运行 ${elapsed} 秒...`);
  }
  throw new Error("任务超过 10 分钟未完成，已停止等待。");
}

function translateJobStage(stage) {
  const text = String(stage || "").trim();
  const stageMap = {
    "Starting research...": "正在启动研究任务...",
    "Running research workflow...": "正在运行研究工作流...",
    "Starting literature analysis...": "正在启动文献分析...",
    "Resolving DOI metadata...": "正在补全文献元数据...",
    "Running LLM literature analysis...": "正在运行文献分析...",
    "Preparing SLR sources...": "正在准备文献综述来源...",
  };
  return stageMap[text] || text;
}

async function readJsonResponse(response) {
  const responseText = await response.text();
  if (!responseText) return {};
  try {
    return JSON.parse(responseText);
  } catch {
    throw new Error(`服务返回了无法解析的数据：${responseText.slice(0, 160)}`);
  }
}

function renderMarkdown(container, markdown) {
  container.innerHTML = markdownToHtml(markdown || "");
}

function markdownToHtml(markdown) {
  const lines = String(markdown).split(/\r?\n/);
  let html = "";
  let inList = false;
  for (const line of lines) {
    if (/^###\s+/.test(line)) {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<h3>${escapeHtml(line.replace(/^###\s+/, ""))}</h3>`;
    } else if (/^##\s+/.test(line)) {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<h2>${escapeHtml(line.replace(/^##\s+/, ""))}</h2>`;
    } else if (/^#\s+/.test(line)) {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<h1>${escapeHtml(line.replace(/^#\s+/, ""))}</h1>`;
    } else if (/^[-*]\s+/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inlineMarkdown(line.replace(/^[-*]\s+/, ""))}</li>`;
    } else if (!line.trim()) {
      if (inList) { html += "</ul>"; inList = false; }
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<p>${inlineMarkdown(line)}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html || '<p class="empty-state">暂无内容。</p>';
}

function inlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function reviewCell(value) {
  return escapeHtml(value || "未说明");
}

function reviewCellWithMeta(value, meta) {
  const body = reviewCell(value);
  return meta ? `${body}<br><small>${escapeHtml(meta)}</small>` : body;
}

function linkOrText(title, source) {
  const cleanTitle = escapeHtml(title || "未命名");
  if (/^https?:\/\//i.test(source || "")) {
    return `<a href="${escapeHtml(source)}" target="_blank" rel="noreferrer">${cleanTitle}</a>`;
  }
  return cleanTitle;
}

function buildDownloadText() {
  const referenceMarkdown = latestReferences.map((reference) => {
    const title = reference.source && /^https?:\/\//i.test(reference.source)
      ? `[${reference.title}](${reference.source})`
      : reference.title;
    return `- ${title} (${reference.branch_name || "来源"}) - ${reference.relevance || ""}`;
  }).join("\n");
  return `${latestMarkdown.trim()}\n\n## 相关文献\n\n${referenceMarkdown}\n`;
}

function appendChatMessage(role, content) {
  const message = document.createElement("div");
  message.className = `chat-message ${role}`;
  message.innerHTML = markdownToHtml(content);
  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function downloadPdfReport() {
  setStatus("正在生成 PDF...");
  try {
    const response = await fetch(apiPath("/api/export/pdf"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: latestTopic || "research-report",
        markdown: buildDownloadText(),
      }),
    });
    if (!response.ok) {
      const payload = await readJsonResponse(response);
    }
    const blob = await response.blob();
    downloadBlob(`${safeFileName(latestTopic)}.pdf`, blob);
    setStatus("PDF 报告已下载。");
  } catch (error) {
    setStatus(`PDF 导出失败：${error.message}`, true);
  }
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatSelectedFilesSummary(files) {
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  const names = files.map((file) => file.name).slice(0, 3).join("、");
  const suffix = files.length > 3 ? `，另有 ${files.length - 3} 个文件` : "";
  return `已添加 ${files.length} 个文件，约 ${formatFileSize(totalSize)}：${names}${suffix}`;
}

function apiPath(path) {
  return `${appBasePath}${path}`;
}

function safeFileName(value) {
  return (value || "research-report")
    .trim()
    .replace(/[<>:"/\\|?*\u0000-\u001f]+/g, "_")
    .replace(/\s+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80) || "research-report";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
