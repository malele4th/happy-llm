const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");

const modeEl = document.getElementById("mode");
const autoDateEl = document.getElementById("auto-date");

function appendMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  wrap.innerHTML = `<div class="bubble">${html}</div>`;
  messagesEl.appendChild(wrap);
  return wrap;
}

function scrollMessageToCenter(element) {
  const containerHeight = messagesEl.clientHeight;
  const elementTop = element.offsetTop;
  const elementHeight = element.offsetHeight;
  const target = elementTop - containerHeight / 2 + elementHeight / 2;
  messagesEl.scrollTop = Math.max(0, target);
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
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
  const userMessage = appendMessage("user", escapeHtml(text));
  scrollMessageToCenter(userMessage);

  const pending = appendMessage("assistant", '<span class="typing">正在检索周报并生成回答…</span>');
  sendBtn.disabled = true;

  const payload = {
    message: text,
    mode: modeEl.value,
    auto_date: autoDateEl.checked,
    k: 5,
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
    scrollToBottom();
  } catch (err) {
    pending.querySelector(".bubble").innerHTML = `<span class="error">${escapeHtml(err.message)}</span>`;
    scrollToBottom();
  } finally {
    sendBtn.disabled = false;
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
