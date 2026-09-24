"use strict";

const view = document.getElementById("view");

async function api(path, method = "GET", body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) throw new Error((data && data.detail) || `HTTP ${res.status}`);
  return data;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function pretty(v) {
  return esc(JSON.stringify(v, null, 2));
}

function badge(label, cls = "") {
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}

function table(headers, rows) {
  const head = headers.map((h) => `<th>${esc(h)}</th>`).join("");
  const body = rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function link(hash, text) {
  return `<a href="#${hash}">${esc(text)}</a>`;
}

// ---- router ----------------------------------------------------------------

const PAGES = {
  overview: overviewPage,
  episodes: episodesPage,
  checks: checksPage,
  tasks: tasksPage,
  rooms: roomsPage,
  benchmarks: benchmarksPage,
  train: trainPage,
};

let pollTimer = null;

async function render() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  const parts = (location.hash.replace(/^#\/?/, "") || "overview").split("/");
  const route = parts[0] || "overview";
  document.querySelectorAll("#nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === route);
  });
  const page = PAGES[route] || overviewPage;
  view.innerHTML = `<div class="loading">Loading…</div>`;
  try {
    await page(parts.slice(1));
  } catch (err) {
    view.innerHTML = `<div class="error">${esc(err.message)}</div>`;
  }
}

window.addEventListener("hashchange", render);
window.addEventListener("DOMContentLoaded", render);

// ---- overview --------------------------------------------------------------

async function overviewPage() {
  const o = await api("/api/overview");
  const tiles = [
    ["Episodes", o.episodes.total], ["Spans", o.spans],
    ["Checks", `${o.checks.enabled}/${o.checks.total} on`],
    ["Tasks", o.tasks], ["Benchmarks", o.benchmarks],
    ["Runs", o.runs], ["Rooms open", o.rooms.open],
  ].map(([k, v]) => `<div class="tile"><div class="num">${esc(v)}</div><div class="lbl">${esc(k)}</div></div>`).join("");

  const total = o.episodes.total || 1;
  const bar = Object.entries(o.episodes.outcomes)
    .map(([label, n]) => `<span class="seg" style="width:${(100 * n) / total}%" title="${esc(label)}: ${n}">${esc(label)}</span>`)
    .join("");

  view.innerHTML = `<h2>Overview</h2>
    <div class="tiles">${tiles}</div>
    <h3>Outcomes</h3>
    <div class="outcome-bar">${bar || `<span class="muted">no episodes yet</span>`}</div>
    <div class="hint">${nextStep(o)}</div>`;
}

function nextStep(o) {
  if (!o.episodes.total)
    return `Capture traces first: add <code>import touchstone; touchstone.trace()</code> to your agent, or run <code>touchstone demo</code>.`;
  if (!o.checks.total)
    return `You have episodes but no checks. ${link("/checks", "Mine or add checks")} to describe what good looks like.`;
  if (!o.checks.enabled)
    return `Checks exist but none are enabled. ${link("/checks", "Enable a check")} so it gates the benchmark.`;
  if (!o.benchmarks)
    return `Ready to prove a model. ${link("/benchmarks", "Create a benchmark")} from your tasks.`;
  if (!o.runs)
    return `Benchmark ready. ${link("/benchmarks", "Run a model")} against it.`;
  return `You're set. Compare runs on the ${link("/benchmarks", "Benchmarks")} page.`;
}

// ---- episodes --------------------------------------------------------------

async function episodesPage(args) {
  if (args[0]) return episodeDetail(args[0]);
  const { episodes, total } = await api("/api/episodes?limit=200");
  const rows = episodes.map((e) => [
    link(`/episodes/${e.id}`, e.name || e.id),
    esc(e.source || ""),
    e.outcome_label ? badge(e.outcome_label, "outcome") : "",
    e.outcome_score == null ? "" : esc(e.outcome_score),
    esc((e.started_at || "").slice(0, 19)),
  ]);
  view.innerHTML = `<h2>Episodes <span class="count">${total}</span></h2>` +
    (rows.length ? table(["name", "source", "outcome", "score", "started"], rows) : `<p class="muted">No episodes captured yet.</p>`);
}

async function episodeDetail(id) {
  const { episode, spans } = await api(`/api/episodes/${id}`);
  const msgs = conversation(spans);
  view.innerHTML = `<div class="crumb">${link("/episodes", "← Episodes")}</div>
    <h2>${esc(episode.name || episode.id)}
      ${episode.outcome_label ? badge(episode.outcome_label, "outcome") : ""}</h2>
    <div class="muted">${esc(episode.source || "")} · ${spans.length} spans</div>
    <div class="chat">${msgs.map(bubble).join("") || `<p class="muted">No LLM turns recorded.</p>`}</div>`;
}

function conversation(spans) {
  const llm = spans.filter((s) => s.kind === "llm");
  if (!llm.length) return [];
  const last = llm[llm.length - 1];
  const msgs = ((last.input && last.input.messages) || []).slice();
  const out = last.output && last.output.message;
  if (out) msgs.push(out);
  return msgs;
}

function bubble(m) {
  const role = m.role || "?";
  const calls = (m.tool_calls || []).map((c) =>
    `<div class="toolcall"><span class="tname">${esc(c.name)}</span><pre>${esc(c.arguments)}</pre></div>`).join("");
  const content = m.content ? `<div class="body">${esc(m.content)}</div>` : "";
  return `<div class="msg ${esc(role)}"><div class="who">${esc(m.name || role)}</div>${content}${calls}</div>`;
}

// ---- checks ----------------------------------------------------------------

async function checksPage() {
  const [{ checks }, { kinds }] = await Promise.all([api("/api/checks"), api("/api/checks/kinds")]);
  const rows = checks.map((c) => [
    `<input type="checkbox" data-enable="${c.id}" ${c.enabled ? "checked" : ""}>`,
    `<code>${esc(c.kind)}</code>`,
    esc(c.name || ""),
    badge(c.severity, c.severity),
    badge(c.source, "src"),
    `<div class="rationale">${esc(c.rationale || "")}</div>`,
    `<button class="ghost" data-try="${c.id}">Try it</button>`,
  ]);
  view.innerHTML = `<h2>Checks <span class="count">${checks.length}</span></h2>
    ${newCheckForm(kinds)}
    <div id="tryDrawer"></div>
    ${checks.length ? table(["on", "kind", "name", "severity", "source", "rationale", ""], rows) : `<p class="muted">No checks yet — mine them or add one above.</p>`}`;

  view.querySelectorAll("[data-enable]").forEach((box) => {
    box.onchange = () => api(`/api/checks/${box.dataset.enable}`, "PATCH", { enabled: box.checked });
  });
  view.querySelectorAll("[data-try]").forEach((b) => {
    b.onclick = () => tryDrawer(b.dataset.try);
  });
  wireNewCheck(kinds);
}

function newCheckForm(kinds) {
  const opts = Object.keys(kinds).map((k) => `<option value="${k}">${k}</option>`).join("");
  return `<details class="panel"><summary>New check</summary>
    <div class="form">
      <label>kind <select id="nc-kind">${opts}</select></label>
      <label>name <input id="nc-name" placeholder="optional"></label>
      <label>severity <select id="nc-sev"><option>hard</option><option>soft</option></select></label>
      <label>applies to <select id="nc-applies"><option>final</option><option>any_turn</option><option>tool_calls</option></select></label>
      <label class="wide">params <textarea id="nc-params" rows="3"></textarea></label>
      <label class="wide">rationale <input id="nc-rationale" placeholder="why this matters"></label>
      <button id="nc-save">Add check</button>
      <span id="nc-msg" class="msg-inline"></span>
    </div></details>`;
}

function wireNewCheck(kinds) {
  const kindSel = document.getElementById("nc-kind");
  const params = document.getElementById("nc-params");
  const hint = () => { params.placeholder = kinds[kindSel.value] || "{}"; };
  kindSel.onchange = hint; hint();
  document.getElementById("nc-save").onclick = async () => {
    const msg = document.getElementById("nc-msg");
    let parsed;
    try {
      parsed = JSON.parse(params.value || "{}");
    } catch (e) {
      msg.textContent = "params is not valid JSON";
      return;
    }
    try {
      await api("/api/checks", "POST", {
        kind: kindSel.value, params: parsed,
        name: document.getElementById("nc-name").value,
        severity: document.getElementById("nc-sev").value,
        applies_to: document.getElementById("nc-applies").value,
        rationale: document.getElementById("nc-rationale").value,
      });
      render();
    } catch (e) {
      msg.textContent = e.message;
    }
  };
}

function tryDrawer(id) {
  const d = document.getElementById("tryDrawer");
  d.innerHTML = `<div class="panel"><strong>Try check <code>${esc(id)}</code></strong>
    <div class="form">
      <label class="wide">output text <textarea id="try-text" rows="3"></textarea></label>
      <label class="wide">tool calls (JSON) <textarea id="try-tools" rows="2" placeholder="[]"></textarea></label>
      <button id="try-run">Evaluate</button>
      <span id="try-result" class="msg-inline"></span>
    </div></div>`;
  document.getElementById("try-run").onclick = async () => {
    const out = document.getElementById("try-result");
    let tools = [];
    const raw = document.getElementById("try-tools").value.trim();
    if (raw) { try { tools = JSON.parse(raw); } catch (e) { out.textContent = "tool calls is not valid JSON"; return; } }
    try {
      const r = await api(`/api/checks/${id}/eval`, "POST", { text: document.getElementById("try-text").value, tool_calls: tools });
      const verdict = r.passed === true ? "PASS" : r.passed === false ? "FAIL" : "N/A";
      out.innerHTML = `<b>${verdict}</b> — ${esc(r.evidence)}`;
    } catch (e) { out.textContent = e.message; }
  };
}

// ---- tasks -----------------------------------------------------------------

async function tasksPage(args) {
  if (args[0]) return taskDetail(args[0]);
  const { tasks, total } = await api("/api/tasks?limit=200");
  const rows = tasks.map((t) => [
    link(`/tasks/${t.id}`, t.name || t.id),
    (t.tags || []).map((x) => badge(x)).join(" "),
    esc(t.check_count),
    esc(t.kind),
  ]);
  view.innerHTML = `<h2>Tasks <span class="count">${total}</span></h2>` +
    (rows.length ? table(["name", "tags", "checks", "kind"], rows) : `<p class="muted">No tasks yet — run mine to cut replay tasks.</p>`);
}

async function taskDetail(id) {
  const t = await api(`/api/tasks/${id}`);
  const ctx = ((t.context || {}).messages || []).map(bubble).join("") || `<p class="muted">no context</p>`;
  const checks = (t.checks || []).map((c) =>
    `<div class="check"><code>${esc(c.kind)}</code> ${esc(c.name || "")} ${badge(c.severity, c.severity)}
      <button class="ghost" data-detach="${c.id}">detach</button></div>`).join("") || `<p class="muted">no checks attached</p>`;
  view.innerHTML = `<div class="crumb">${link("/tasks", "← Tasks")}</div>
    <h2>${esc(t.name || t.id)} ${(t.tags || []).map((x) => badge(x)).join(" ")}</h2>
    <button id="interview">Interview</button>
    <h3>Context</h3><div class="chat">${ctx}</div>
    <h3>Reference</h3><pre>${pretty(t.reference)}</pre>
    <h3>Checks</h3>${checks}`;

  view.querySelectorAll("[data-detach]").forEach((b) => {
    b.onclick = async () => { await api(`/api/tasks/${id}/checks/${b.dataset.detach}`, "DELETE"); taskDetail(id); };
  });
  document.getElementById("interview").onclick = async () => {
    const room = await api("/api/rooms", "POST", { task_id: id, topic: `review of ${t.name}` });
    location.href = `/rooms/${room.room.id}`;
  };
}

// ---- rooms -----------------------------------------------------------------

async function roomsPage() {
  const { rooms } = await api("/api/rooms");
  const rows = rooms.map((r) => [
    `<a href="/rooms/${r.id}">${esc(r.topic)}</a>`,
    r.task_name ? link(`/tasks/${r.task_id}`, r.task_name) : "",
    r.closed_at ? badge("closed") : badge("open", "outcome"),
  ]);
  view.innerHTML = `<h2>Rooms <span class="count">${rooms.length}</span></h2>` +
    (rows.length ? table(["topic", "task", "state"], rows) : `<p class="muted">No interview rooms yet — start one from a task.</p>`);
}

// ---- benchmarks ------------------------------------------------------------

async function benchmarksPage() {
  const [{ benchmarks }, { runs }, { tasks }] = await Promise.all([
    api("/api/benchmarks"), api("/api/runs"), api("/api/tasks?limit=1"),
  ]);
  const benchRows = benchmarks.map((b) => [
    esc(b.name), esc((b.task_ids || []).length), runForm(b),
  ]);
  const runRows = runs.map((r) => [
    `<code>${esc(r.id.slice(-8))}</code>`, esc(r.model_spec),
    r.finished_at ? badge("done", "outcome") : `<span data-run="${r.id}">${r.done}/${r.total}</span>`,
  ]);
  view.innerHTML = `<h2>Benchmarks</h2>
    <details class="panel"><summary>New benchmark</summary>
      <div class="form">
        <label>name <input id="bm-name"></label>
        <label>tags (comma-sep, blank = all) <input id="bm-tags" placeholder="failure"></label>
        <button id="bm-save">Create</button><span id="bm-msg" class="msg-inline"></span>
      </div></details>
    ${benchmarks.length ? table(["benchmark", "tasks", "run a model"], benchRows) : `<p class="muted">No benchmarks — create one${tasks.total ? "" : " after mining tasks"}.</p>`}
    <h3>Runs</h3>
    ${runs.length ? table(["run", "model", "progress"], runRows) : `<p class="muted">No runs yet.</p>`}
    ${proofPanel(runs)}`;

  document.getElementById("bm-save").onclick = createBenchmark;
  view.querySelectorAll("[data-runbm]").forEach((f) => { f.onsubmit = startRun; });
  document.getElementById("proof-go").onclick = showProof;
  pollRuns(runs);
}

function runForm(b) {
  return `<form data-runbm="${b.id}" class="runform">
    <input name="model" placeholder="scripted" required>
    <input name="conc" type="number" min="1" value="4" title="concurrency">
    <button>Run</button></form>`;
}

async function createBenchmark() {
  const msg = document.getElementById("bm-msg");
  const tags = document.getElementById("bm-tags").value.split(",").map((s) => s.trim()).filter(Boolean);
  const body = { name: document.getElementById("bm-name").value };
  if (tags.length) body.tags = tags; else body.all = true;
  try { await api("/api/benchmarks", "POST", body); render(); }
  catch (e) { msg.textContent = e.message; }
}

async function startRun(ev) {
  ev.preventDefault();
  const form = ev.currentTarget;
  const body = {
    benchmark: form.dataset.runbm,
    model_spec: form.model.value.trim(),
    concurrency: Number(form.conc.value) || 4,
  };
  try { await api("/api/runs", "POST", body); render(); }
  catch (e) { alert(e.message); }
}

function pollRuns(runs) {
  const pending = runs.filter((r) => !r.finished_at);
  if (!pending.length) return;
  pollTimer = setInterval(async () => {
    let done = true;
    for (const r of pending) {
      const cur = await api(`/api/runs/${r.id}`);
      const cell = view.querySelector(`[data-run="${r.id}"]`);
      if (cell) cell.textContent = `${cur.done}/${cur.total}`;
      if (!cur.finished_at) done = false;
    }
    if (done) render();
  }, 2000);
}

function proofPanel(runs) {
  if (runs.length < 2) return "";
  const opts = runs.map((r) => `<option value="${r.id}">${esc(r.model_spec)} (${esc(r.id.slice(-6))})</option>`).join("");
  return `<h3>Proof</h3><div class="form">
    <label>candidate <select id="proof-cand">${opts}</select></label>
    <label>incumbent <select id="proof-inc">${opts}</select></label>
    <button id="proof-go">Compare</button></div><div id="proof-out"></div>`;
}

// ---- train -----------------------------------------------------------------

async function trainPage() {
  const { benchmarks } = await api("/api/benchmarks");
  if (!benchmarks.length) {
    view.innerHTML = `<h2>Train</h2><p class="muted">No benchmarks yet — create one on the Benchmarks page.</p>`;
    return;
  }
  const opts = benchmarks.map((b) =>
    `<option value="${b.id}">${esc(b.name)} (${(b.task_ids || []).length} tasks)</option>`).join("");
  view.innerHTML = `<h2>Train</h2>
    <p class="muted">Prepare SFT / preference / RL datasets from a benchmark, then emit a runnable backend config. Nothing trains here — Touchstone hands you the datasets and the exact command to run on GPU infra.</p>
    <div class="form">
      <label>benchmark <select id="tr-bench">${opts}</select></label>
      <button id="tr-prepare">Prepare datasets</button>
    </div>
    <div id="tr-out"></div>`;
  document.getElementById("tr-prepare").onclick = trainPrepare;
}

async function trainPrepare() {
  const bench = document.getElementById("tr-bench").value;
  const out = document.getElementById("tr-out");
  out.innerHTML = `<div class="loading">Preparing…</div>`;
  try {
    const b = await api("/api/train/prepare", "POST", { benchmark: bench });
    out.innerHTML = trainResult(bench, b);
    wireTrainSubmit(bench);
  } catch (e) {
    out.innerHTML = `<div class="error">${esc(e.message)}</div>`;
  }
}

function dlLink(bench, file, rows) {
  const href = `/api/train/download?benchmark=${encodeURIComponent(bench)}&file=${file}`;
  const count = rows === undefined ? "" : ` <span class="muted">(${rows} rows)</span>`;
  return `<a href="${href}" download>${file}</a>${count}`;
}

function trainResult(bench, b) {
  const c = b.counts;
  return `<div class="panel">
    <p>Wrote datasets to <code>${esc(b.out_dir)}</code></p>
    <ul>
      <li>${dlLink(bench, "sft.jsonl", c.sft)}</li>
      <li>${dlLink(bench, "preference.jsonl", c.preference)}</li>
      <li>${dlLink(bench, "rl_tasks.jsonl", c.rl_tasks)}</li>
      <li>${dlLink(bench, "manifest.json")}</li>
    </ul>
    <div class="form">
      <label>backend <select id="tr-backend">
        <option value="null">null (write plan)</option>
        <option value="art">art (RL)</option>
        <option value="trl">trl (SFT)</option>
      </select></label>
      <label>base model <input id="tr-model" placeholder="Qwen/Qwen2.5-7B-Instruct"></label>
      <button id="tr-submit">Submit</button>
    </div>
    <div id="tr-submit-out"></div>
  </div>`;
}

function wireTrainSubmit(bench) {
  document.getElementById("tr-submit").onclick = async () => {
    const out = document.getElementById("tr-submit-out");
    const body = { benchmark: bench, backend: document.getElementById("tr-backend").value };
    const model = document.getElementById("tr-model").value.trim();
    if (model) body.base_model = model;
    out.innerHTML = `<div class="loading">Submitting…</div>`;
    try {
      const r = await api("/api/train/submit", "POST", body);
      out.innerHTML = `<p class="muted"><b>${esc(r.status)}</b> — ${esc(r.detail)}</p>`;
    } catch (e) {
      out.innerHTML = `<div class="error">${esc(e.message)}</div>`;
    }
  };
}

async function showProof() {
  const cand = document.getElementById("proof-cand").value;
  const inc = document.getElementById("proof-inc").value;
  const out = document.getElementById("proof-out");
  try {
    const p = await api(`/api/proof?candidate=${cand}&incumbent=${inc}`);
    const verdict = { both_pass: "= both", only_incumbent: "− lost", only_candidate: "+ gained", both_fail: "× both fail" };
    const rows = p.tasks.map((r) => [
      esc(r.name), r.incumbent_passed ? "pass" : "fail", r.candidate_passed ? "pass" : "fail", verdict[r.category],
    ]);
    const c = p.counts;
    out.innerHTML = `<p class="muted">both pass ${c.both_pass} · only incumbent ${c.only_incumbent} · only candidate ${c.only_candidate} · both fail ${c.both_fail}</p>` +
      table(["task", "incumbent", "candidate", "verdict"], rows);
  } catch (e) { out.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
}
