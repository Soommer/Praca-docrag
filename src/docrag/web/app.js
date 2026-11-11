// docrag - frontend: logowanie, pytania (/api/ask/stream, Server-Sent Events) i dokumenty użytkownika.
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

let me = { user: null, is_admin: false, auth: "local" };

// ---------------------------------------------------------------- wspólne

// fetch z obsługą wygasłej sesji: 401 -> ekran logowania
async function api(url, options = {}) {
  const r = await fetch(url, options);
  if (r.status === 401 && me.auth === "local") {
    showLogin();
    throw new Error("Sesja wygasła - zaloguj się ponownie.");
  }
  return r;
}

async function errorOf(r) {
  const body = await r.json().catch(() => ({}));
  if (Array.isArray(body.detail)) return "Nieprawidłowe dane.";
  return body.detail || `Błąd serwera (${r.status})`;
}

function showError(box, msg) {
  box.textContent = msg || "";
  box.hidden = !msg;
}

async function checkHealth() {
  const status = $("status");
  try {
    const h = await (await fetch("api/health")).json();
    const llm = h.llm.backend === "none" ? "bez LLM" : `LLM: ${h.llm.model || h.llm.backend}`;
    status.textContent = h.qdrant.ok ? `${h.qdrant.points} fragmentów w bazie · ${llm}` : "baza niedostępna";
    status.className = "status " + (h.status === "ok" ? "ok" : "bad");
  } catch {
    status.textContent = "serwer niedostępny";
    status.className = "status bad";
  }
}

// ---------------------------------------------------------------- logowanie

function showLogin() {
  $("app-view").hidden = true;
  $("account").hidden = true;
  $("login-view").hidden = false;
  $("login-user").focus();
}

function showApp() {
  $("login-view").hidden = true;
  $("app-view").hidden = false;
  document.body.classList.toggle("is-admin", me.is_admin);
  $("account").hidden = !me.user;
  $("whoami").replaceChildren(el("b", { textContent: me.user || "" }), me.is_admin ? " (admin)" : "");
  $("logout").hidden = me.auth !== "local";
  $("docs-tab-button").hidden = !me.user;          // bez logowania (api.auth: none) nie ma własnych dokumentów
  $("docs-heading").textContent = me.is_admin ? "Wszystkie dokumenty" : "Moje dokumenty";
  $("question").focus();
}

async function loadSession() {
  me = await (await fetch("api/me")).json();
  if (me.user || me.auth === "none") showApp();
  else showLogin();
}

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError($("login-error"), "");
  const r = await fetch("api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: $("login-user").value, password: $("login-pass").value }),
  });
  if (!r.ok) return showError($("login-error"), await errorOf(r));
  $("login-pass").value = "";
  await loadSession();
});

$("logout").addEventListener("click", async () => {
  await fetch("api/logout", { method: "POST" });
  me = { user: null, is_admin: false, auth: me.auth };
  $("result").hidden = true;
  $("docs-body").replaceChildren();
  showLogin();
});

// ---------------------------------------------------------------- zakładki

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t === tab);
    for (const p of document.querySelectorAll(".tab-panel")) p.hidden = p.id !== tab.dataset.tab;
    if (tab.dataset.tab === "docs-tab") loadDocuments();
  });
}

// ---------------------------------------------------------------- pytania

const answerBox = $("answer");
const sourcesList = $("sources");

// Odpowiedź z odwołaniami [n] zamienionymi na linki do źródeł (bez innerHTML - tekst z modelu nie jest HTML-em).
function renderAnswer(text, sourceCount) {
  answerBox.replaceChildren();
  let last = 0;
  for (const m of text.matchAll(/\[(\d+)\]/g)) {
    const n = Number(m[1]);
    if (n < 1 || n > sourceCount) continue;
    answerBox.append(text.slice(last, m.index));
    const a = el("a", { href: `#src-${n}`, className: "cite", textContent: m[0] });
    a.addEventListener("click", () => { $(`src-${n}`).querySelector("details").open = true; });
    answerBox.append(a);
    last = m.index + m[0].length;
  }
  answerBox.append(text.slice(last));
}

function renderSources(sources) {
  sourcesList.replaceChildren();
  for (const s of sources) {
    const parts = [s.related_to ? "źródło pomocnicze (przywołane w innym)" : `trafność ${s.score.toFixed(2)}`];
    if (s.version) parts.push(s.version);
    if (s.section_path.length) parts.push(s.section_path.at(-1));
    const summary = el("summary", {},
      el("span", { className: "num", textContent: `[${s.n}] ` }),
      `${s.title}${s.unit ? " - " + s.unit : ""}`,
      el("span", { className: "meta", textContent: parts.join(" · ") }));
    const details = el("details", {}, summary, el("p", { className: "text", textContent: s.text }));
    sourcesList.append(el("li", { id: `src-${s.n}` }, details));
  }
  if (!sources.length) {
    sourcesList.append(el("li", { className: "meta", textContent: "Brak źródeł.", style: "padding: 10px 14px" }));
  }
}

function renderStats(stats) {
  const s = (ms) => (ms / 1000).toFixed(1) + " s";
  const total = ["ms_embed", "ms_search", "ms_rerank", "ms_expand", "ms_llm"]
    .reduce((sum, k) => sum + (stats[k] || 0), 0);
  $("stats").textContent = `czas: ${s(total)} (wyszukiwanie ${s(total - (stats.ms_llm || 0))}` +
    (stats.ms_llm ? `, model ${s(stats.ms_llm)})` : ")");
}

async function ask(q) {
  showError($("error"), "");
  $("result").hidden = false;
  answerBox.className = "answer loading";
  answerBox.textContent = "Szukam w dokumentach…";
  $("stats").textContent = "";
  sourcesList.replaceChildren();

  const r = await api("api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q }),
  });
  if (!r.ok) throw new Error(await errorOf(r));

  let sources = [];
  let answer = "";
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const event = /^event: (.*)$/m.exec(raw)?.[1];
      const data = JSON.parse(/^data: (.*)$/m.exec(raw)?.[1] ?? "null");
      if (event === "sources") {
        sources = data;
        renderSources(sources);
        answerBox.textContent = "Piszę odpowiedź…";
      } else if (event === "token") {
        if (!answer) answerBox.className = "answer";
        answer += data.text;
        answerBox.textContent = answer;
      } else if (event === "done") {
        renderStats(data.stats);
      } else if (event === "error") {
        throw new Error(data.detail);
      }
    }
  }
  answerBox.className = "answer";
  renderAnswer(answer, sources.length);
}

$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (!q) return;
  $("submit").disabled = true;
  try {
    await ask(q);
  } catch (err) {
    showError($("error"), err.message || "Nie udało się połączyć z serwerem.");
    if (answerBox.classList.contains("loading")) $("result").hidden = true;
  } finally {
    $("submit").disabled = false;
  }
});

$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
});

// ---------------------------------------------------------------- dokumenty

const VISIBILITY = { private: "tylko właściciel", public: "wszyscy", restricted: "wybrane grupy" };
let refreshTimer = null;

function statusCell(d) {
  if (d.status === "ready") return el("td", {}, el("span", { className: "badge ready", textContent: "gotowy" }),
    el("span", { className: "sub", textContent: `${d.chunks} fragmentów` }));
  if (d.status === "processing") return el("td", {}, el("span", { className: "badge processing", textContent: "przetwarzanie…" }));
  return el("td", {}, el("span", { className: "badge error", textContent: "błąd" }),
    el("span", { className: "sub", textContent: d.error || "" }));
}

function renderDocuments(docs) {
  const body = $("docs-body");
  body.replaceChildren();
  for (const d of docs) {
    const del = d.can_delete ? el("button", { type: "button", className: "danger", textContent: "Usuń" }) : "";
    if (del) del.addEventListener("click", () => deleteDocument(d));
    body.append(el("tr", {},
      el("td", {}, d.title, el("span", { className: "sub", textContent: d.filename || d.doc_id })),
      el("td", { className: "admin-only", textContent: d.owner || "system (CLI)" }),
      el("td", { textContent: VISIBILITY[d.visibility] || d.visibility }),
      statusCell(d),
      el("td", { textContent: new Date(d.created_at).toLocaleString("pl-PL", { dateStyle: "short", timeStyle: "short" }) }),
      el("td", {}, del)));
  }
  $("docs-table").hidden = !docs.length;
  $("docs-empty").hidden = docs.length > 0;
}

async function loadDocuments() {
  clearTimeout(refreshTimer);
  try {
    const r = await api("api/documents");
    if (!r.ok) throw new Error(await errorOf(r));
    const docs = await r.json();
    renderDocuments(docs);
    // dopóki coś się przetwarza - odświeżaj co 3 s
    if (docs.some((d) => d.status === "processing")) refreshTimer = setTimeout(loadDocuments, 3000);
  } catch (err) {
    showError($("upload-error"), err.message);
  }
}

async function deleteDocument(d) {
  const whose = d.owner && d.owner !== me.user ? ` użytkownika ${d.owner}` : d.owner ? "" : " (systemowy)";
  if (!confirm(`Usunąć dokument „${d.title}”${whose}? Tej operacji nie można cofnąć.`)) return;
  const r = await api(`api/documents/${encodeURIComponent(d.doc_id)}`, { method: "DELETE" });
  if (!r.ok) showError($("upload-error"), await errorOf(r));
  await loadDocuments();
  checkHealth();
}

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError($("upload-error"), "");
  const file = $("upload-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("title", $("upload-title").value);
  form.append("public", $("upload-public").checked);
  $("upload-submit").disabled = true;
  try {
    const r = await api("api/documents", { method: "POST", body: form });
    if (!r.ok) throw new Error(await errorOf(r));
    $("upload-form").reset();
    await loadDocuments();
  } catch (err) {
    showError($("upload-error"), err.message);
  } finally {
    $("upload-submit").disabled = false;
  }
});

// ---------------------------------------------------------------- start

checkHealth();
loadSession().catch(() => showError($("login-error"), "Nie udało się połączyć z serwerem."));
