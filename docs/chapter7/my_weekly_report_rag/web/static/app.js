const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");

const modeEl = document.getElementById("mode");
const autoDateEl = document.getElementById("auto-date");
const yearEl = document.getElementById("year");
const monthEl = document.getElementById("month");
const tokenEl = document.getElementById("token");

const savedToken = localStorage.getItem("weekly_rag_token");
if (savedToken) {
  tokenEl.value = savedToken;
}

tokenEl.addEventListener("change", () => {
  localStorage.setItem("weekly_rag_token", tokenEl.value.trim());
});

function appendMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  wrap.innerHTML = `<div class="bubble">${html}</div>`;
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function renderCitations(citations) {
  if (!citations.length) return "";
  const items = citations
    .map(
      (c) =>
        `<div class="citation"><strong>[${c.index}]</strong> ${c.date} · ${c.project} · score ${c.score.toFixed(3)}<br>${escapeHtml(c.preview)}</div>`
    )
    .join("");
  return `<div class="citations"><h3>引用来源</h3>${items}</div>`;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderFilterMeta(data) {
  const parts = [`模式: ${data.mode}`];
  if (data.filter_year) {
    parts.push(`过滤: ${data.filter_year}年${data.filter_month ? data.filter_month + "月" : ""}`);
  }
  return `<div class="meta">${parts.join(" · ")}</div>`;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    statusText.textContent = `已连接 · ${data.chunk_count} 条周报片段`;
  } catch {
    statusText.textContent = "服务未连接";
  }
}

async function sendMessage(text) {
  appendMessage("user", escapeHtml(text));
  const pending = appendMessage("assistant", '<span class="typing">正在检索周报并生成回答…</span>');
  sendBtn.disabled = true;

  const headers = { "Content-Type": "application/json" };
  const token = tokenEl.value.trim();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const payload = {
    message: text,
    mode: modeEl.value,
    auto_date: autoDateEl.checked,
    k: 5,
  };
  const year = yearEl.value.trim();
  const month = monthEl.value.trim();
  if (year) payload.year = Number(year);
  if (month) payload.month = Number(month);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "请求失败");
    }

    const html =
      escapeHtml(data.answer) +
      renderCitations(data.citations) +
      renderFilterMeta(data);
    pending.querySelector(".bubble").innerHTML = html;
  } catch (err) {
    pending.querySelector(".bubble").innerHTML = `<span class="error">${escapeHtml(err.message)}</span>`;
  } finally {
    sendBtn.disabled = false;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  sendMessage(text);
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

document.querySelectorAll(".hint").forEach((btn) => {
  btn.addEventListener("click", () => {
    inputEl.value = btn.textContent.trim();
    inputEl.focus();
  });
});

checkHealth();
