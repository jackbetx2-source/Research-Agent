const form = document.querySelector("#researchForm");
const topicInput = document.querySelector("#topicInput");
const fastMode = document.querySelector("#fastMode");
const runButton = document.querySelector("#runButton");
const copyButton = document.querySelector("#copyButton");
const statusText = document.querySelector("#statusText");
const resultOutput = document.querySelector("#resultOutput");
const pageViews = document.querySelectorAll("[data-page]");
const pageButtons = document.querySelectorAll("[data-page-target]");
const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const chatMessages = document.querySelector("#chatMessages");
const chatStatus = document.querySelector("#chatStatus");
const chatSendButton = document.querySelector("#chatSendButton");

let latestMarkdown = "";
let chatHistory = [];
const appBasePath = window.location.pathname.startsWith("/v2") ? "/v2" : "";

initPageNavigation();

if (window.location.protocol === "file:") {
  setStatus("请先双击 start_web.bat 或运行 python web_app.py 8000，再打开 http://127.0.0.1:8000。", true);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const topic = topicInput.value.trim();
  if (!topic) return;

  if (window.location.protocol === "file:") {
    setStatus("直接打开 HTML 只能查看界面，不能运行 research。请通过本地服务访问。", true);
    return;
  }

  setRunning(true);
  setStatus("正在研究，通常需要 20-90 秒...");
  renderMarkdown("## 正在生成\n\n请稍等，最终报告会显示在这里。");

  try {
    const response = await fetch(apiPath("/api/research"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, fast: fastMode.checked }),
    });
    let payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }

    if (payload.job_id) {
      payload = await waitForResearchJob(payload.job_id);
    }

    if (!payload.final_report) {
      throw new Error("Research finished without a report. Check server logs.");
    }

    latestMarkdown = payload.final_report;
    renderMarkdown(payload.final_report);
    setStatus(`完成，已保存到 ${payload.output_path}`);
    copyButton.disabled = false;
  } catch (error) {
    latestMarkdown = "";
    renderMarkdown(`## 运行失败\n\n${error.message}`);
    setStatus("运行失败，请检查本地服务或模型配置。", true);
    copyButton.disabled = true;
  } finally {
    setRunning(false);
  }
});

copyButton.addEventListener("click", async () => {
  if (!latestMarkdown) return;
  await navigator.clipboard.writeText(latestMarkdown);
  setStatus("结果已复制。");
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  if (window.location.protocol === "file:") {
    setChatStatus("请通过本地服务访问后再提问。", true);
    return;
  }

  appendChatMessage("user", message);
  chatHistory.push({ role: "user", content: message });
  chatInput.value = "";
  setChatRunning(true);
  setChatStatus("LLM 正在回答...");

  try {
    const response = await fetch(apiPath("/api/chat"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: chatHistory.slice(-8) }),
    });
    const payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }

    const answer = payload.answer || "我没有收到有效回答。";
    appendChatMessage("assistant", answer);
    chatHistory.push({ role: "assistant", content: answer });
    setChatStatus("已回答。");
  } catch (error) {
    appendChatMessage("assistant", `抱歉，回答失败：${error.message}`);
    setChatStatus("回答失败，请检查本地服务或模型配置。", true);
  } finally {
    setChatRunning(false);
  }
});

function initPageNavigation() {
  const initialPage = window.location.hash === "#about" ? "about" : "workspace";
  showPage(initialPage, false);

  pageButtons.forEach((button) => {
    button.addEventListener("click", () => {
      showPage(button.dataset.pageTarget);
    });
  });

  window.addEventListener("hashchange", () => {
    const page = window.location.hash === "#about" ? "about" : "workspace";
    showPage(page, false);
  });
}

function showPage(pageName, updateHash = true) {
  const nextPage = pageName === "about" ? "about" : "workspace";

  pageViews.forEach((view) => {
    view.classList.toggle("is-active", view.dataset.page === nextPage);
  });

  pageButtons.forEach((button) => {
    const isActive = button.dataset.pageTarget === nextPage;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-current", isActive ? "page" : "false");
  });

  if (updateHash) {
    const nextHash = nextPage === "about" ? "#about" : "#workspace";
    if (window.location.hash !== nextHash) {
      history.pushState(null, "", nextHash);
    }
  }
}

function setRunning(isRunning) {
  runButton.disabled = isRunning;
  topicInput.disabled = isRunning;
  fastMode.disabled = isRunning;
  runButton.textContent = isRunning ? "研究中..." : "开始研究";
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

function setChatStatus(message, isError = false) {
  chatStatus.textContent = message;
  chatStatus.classList.toggle("error", isError);
}

function renderMarkdown(markdown) {
  resultOutput.innerHTML = markdownToHtml(markdown);
}

function appendChatMessage(role, content) {
  const message = document.createElement("div");
  message.className = `chat-message ${role}`;
  message.innerHTML = markdownToHtml(content);
  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function waitForResearchJob(jobId) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < 10 * 60 * 1000) {
    await sleep(2000);
    const response = await fetch(apiPath(`/api/research/${jobId}`), { cache: "no-store" });
    const payload = await readJsonResponse(response);

    if (!response.ok) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }

    if (payload.status === "done") return payload;
    if (payload.status === "error") throw new Error(payload.error || "Research failed.");

    const elapsed = Math.floor((Date.now() - startedAt) / 1000);
    setStatus(`正在研究，已运行 ${elapsed} 秒...`);
  }

  throw new Error("Research timed out after 10 minutes.");
}

async function readJsonResponse(response) {
  const responseText = await response.text();
  if (!responseText) return {};

  try {
    return JSON.parse(responseText);
  } catch {
    throw new Error(responseText.slice(0, 300) || `HTTP ${response.status}`);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function apiPath(path) {
  return `${appBasePath}${path}`;
}

function markdownToHtml(markdown) {
  const lines = markdown.split(/\r?\n/);
  let html = "";
  let listOpen = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      if (listOpen) {
        html += "</ul>";
        listOpen = false;
      }
      continue;
    }

    if (trimmed.startsWith("### ")) {
      if (listOpen) {
        html += "</ul>";
        listOpen = false;
      }
      html += `<h3>${inline(trimmed.slice(4))}</h3>`;
    } else if (trimmed.startsWith("## ")) {
      if (listOpen) {
        html += "</ul>";
        listOpen = false;
      }
      html += `<h2>${inline(trimmed.slice(3))}</h2>`;
    } else if (trimmed.startsWith("# ")) {
      if (listOpen) {
        html += "</ul>";
        listOpen = false;
      }
      html += `<h1>${inline(trimmed.slice(2))}</h1>`;
    } else if (/^[-*]\s+/.test(trimmed)) {
      if (!listOpen) {
        html += "<ul>";
        listOpen = true;
      }
      html += `<li>${inline(trimmed.replace(/^[-*]\s+/, ""))}</li>`;
    } else {
      if (listOpen) {
        html += "</ul>";
        listOpen = false;
      }
      html += `<p>${inline(trimmed)}</p>`;
    }
  }

  if (listOpen) html += "</ul>";
  return html || '<p class="empty-state">结果会显示在这里。</p>';
}

function inline(text) {
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
