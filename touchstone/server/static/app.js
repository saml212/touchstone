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
    ["Tasks", o.tasks.total], ["Benchmarks", o.benchmarks],
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
    <div class="hint">${esc(o.next_step)}</div>`;
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
  const llm = spans.filter((s) => s.kind === "model");
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

function checkCell(c) {
  // Rule and reason read first; the machine params (kind + values) sit underneath, muted.
  const rule = c.rule ? `<div class="crule">${esc(c.rule)}</div>` : "";
  const because = c.because ? `<div class="rationale">${esc(c.because)}</div>` : "";
  return `<div class="cname">${esc(c.name || c.kind)}</div>${rule}${because}` +
    `<div class="cmeta"><code>${esc(c.kind)}</code> ${esc(JSON.stringify(c.params))}</div>`;
}

async function checksPage() {
  const [{ checks }, { kinds }] = await Promise.all([api("/api/checks"), api("/api/checks/kinds")]);
  const rows = checks.map((c) => [
    `<input type="checkbox" data-enable="${esc(c.name)}" ${c.enabled ? "checked" : ""}>`,
    checkCell(c),
    badge(c.severity, c.severity),
    badge(c.source, "src"),
    `<button class="ghost" data-try="${esc(c.name)}">Try it</button>`,
  ]);
  view.innerHTML = `<h2>Checks <span class="count">${checks.length}</span></h2>
    ${newCheckForm(kinds)}
    <div id="tryDrawer"></div>
    ${checks.length ? table(["on", "check", "severity", "source", ""], rows) : `<p class="muted">No checks yet — mine them or add one above.</p>`}`;

  view.querySelectorAll("[data-enable]").forEach((box) => {
    box.onchange = () => api(`/api/checks/${encodeURIComponent(box.dataset.enable)}`, "PATCH", { enabled: box.checked });
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
      <label class="wide">because <input id="nc-rationale" placeholder="why this matters"></label>
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
        because: document.getElementById("nc-rationale").value,
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
      const r = await api(`/api/checks/${encodeURIComponent(id)}/eval`, "POST", { text: document.getElementById("try-text").value, tool_calls: tools });
      const verdict = r.passed === true ? "PASS" : r.passed === false ? "FAIL" : "N/A";
      out.innerHTML = `<b>${verdict}</b> — ${esc(r.evidence)}`;
    } catch (e) { out.textContent = e.message; }
  };
}

// ---- tasks -----------------------------------------------------------------

const TASK_QUEUES = [
  ["active", "ready for benchmarks", ""],
  ["needs_checks", "an empty reply already passes — add a check that measures the work", "Interview"],
  ["needs_solution", "the recorded reply fails its own checks — supply a passing one", "Ask teacher"],
];

async function tasksPage(args) {
  if (args[0]) return taskDetail(args[0]);
  const { tasks, total } = await api("/api/tasks?limit=500");
  if (!total) {
    view.innerHTML = `<h2>Tasks</h2><p class="muted">No tasks yet — run mine to cut replay tasks.</p>`;
    return;
  }
  const sections = TASK_QUEUES.map(([status, hint, action]) => {
    const group = tasks.filter((t) => t.status === status);
    const rows = group.map((t) => [
      link(`/tasks/${t.name}`, t.name),
      (t.tags || []).map((x) => badge(x)).join(" "),
      esc(t.check_count),
      action ? `<button class="ghost" data-${status}="${esc(t.name)}">${action}</button>` : "",
    ]);
    return `<h3>${status} <span class="count">${group.length}</span></h3>
      <p class="muted">${hint}</p>` +
      (rows.length ? table(["name", "tags", "checks", ""], rows) : `<p class="muted">none</p>`);
  }).join("");
  view.innerHTML = `<h2>Tasks <span class="count">${total}</span></h2>${sections}`;
  wireTaskActions();
}

function wireTaskActions() {
  view.querySelectorAll("[data-needs_checks]").forEach((b) => {
    b.onclick = () => startInterview(b.dataset.needs_checks);
  });
  view.querySelectorAll("[data-needs_solution]").forEach((b) => {
    b.onclick = async () => {
      b.disabled = true;
      b.textContent = "asking…";
      try {
        const r = await api(`/api/tasks/${encodeURIComponent(b.dataset.needs_solution)}/teach`, "POST", {});
        if (r.accepted && r.status === "active") render();
        else { b.disabled = false; b.textContent = "Ask teacher"; alert(`Teacher (${r.teacher}) could not produce a passing reply.`); }
      } catch (e) { b.disabled = false; b.textContent = "Ask teacher"; alert(e.message); }
    };
  });
}

async function startInterview(taskName) {
  const room = await api("/api/rooms", "POST", { task_id: taskName, topic: `review of ${taskName}` });
  location.href = `/rooms/${room.room.id}`;
}

function tomlBlock(block) {
  // Render a check as it reads in task.toml: the block header, then key = value per line.
  const lines = ["[[metadata.touchstone.check]]"];
  for (const [k, v] of Object.entries(block || {})) lines.push(`${k} = ${JSON.stringify(v)}`);
  return lines.join("\n");
}

async function taskDetail(id) {
  const t = await api(`/api/tasks/${id}`);
  const ctx = ((t.context || {}).messages || []).map(bubble).join("") || `<p class="muted">no context</p>`;
  const checks = (t.checks || []).map((c) =>
    `<div class="check-block"><button class="ghost" data-detach="${esc(c.name)}">detach</button>
      <pre>${esc(tomlBlock(c.block))}</pre></div>`).join("") || `<p class="muted">no checks attached</p>`;
  view.innerHTML = `<div class="crumb">${link("/tasks", "← Tasks")}</div>
    <h2>${esc(t.name)} ${(t.tags || []).map((x) => badge(x)).join(" ")}</h2>
    <button id="interview">Interview</button>
    <h3>Context</h3><div class="chat">${ctx}</div>
    <h3>Reference</h3><pre>${pretty(t.reference)}</pre>
    <h3>Checks</h3>${checks}`;

  view.querySelectorAll("[data-detach]").forEach((b) => {
    b.onclick = async () => { await api(`/api/tasks/${id}/checks/${encodeURIComponent(b.dataset.detach)}`, "DELETE"); taskDetail(id); };
  });
  document.getElementById("interview").onclick = async () => {
    const room = await api("/api/rooms", "POST", { task_id: t.name, topic: `review of ${t.name}` });
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
    esc(b.name), esc(b.task_count), loopForm(b),
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
    ${benchmarks.length ? table(["benchmark", "tasks", "run / sample / distill"], benchRows) : `<p class="muted">No benchmarks — create one${tasks.total ? "" : " after mining tasks"}.</p>`}
    <div id="loop-out"></div>
    <h3>Runs</h3>
    ${runs.length ? table(["run", "model", "progress"], runRows) : `<p class="muted">No runs yet.</p>`}
    ${proofPanel(runs)}`;

  document.getElementById("bm-save").onclick = createBenchmark;
  wireLoopForms();
  document.getElementById("proof-go").onclick = showProof;
  pollRuns(runs);
}

function loopForm(b) {
  return `<form data-bench="${esc(b.name)}" class="runform">
    <input name="model" placeholder="scripted" required title="student / candidate model">
    <input name="teacher" placeholder="teacher (optional)" title="teacher spec">
    <input name="variants" type="number" min="1" value="1" title="variants per failing task">
    <button data-act="run">Run</button>
    <button data-act="sample">Sample</button>
    <button data-act="distill">Distill</button>
  </form>`;
}

function wireLoopForms() {
  view.querySelectorAll("form[data-bench]").forEach((form) => {
    form.querySelectorAll("button[data-act]").forEach((btn) => {
      btn.onclick = (ev) => { ev.preventDefault(); loopAction(form, btn.dataset.act); };
    });
  });
}

async function loopAction(form, act) {
  const bench = form.dataset.bench;
  const model = form.model.value.trim();
  if (!model) { alert("enter a student / model spec"); return; }
  if (act === "run") return startRun2(bench, model);
  const out = document.getElementById("loop-out");
  out.innerHTML = `<div class="loading">${act === "sample" ? "Sampling" : "Distilling"}…</div>`;
  const body = { target: bench, student: model, teacher: form.teacher.value.trim() || undefined };
  if (act === "sample") body.variants = Number(form.variants.value) || 1;
  try {
    const r = await api(`/api/${act}`, "POST", body);
    out.innerHTML = act === "sample" ? sampleResult(r) : distillResult(r);
  } catch (e) { out.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
}

async function startRun2(bench, model) {
  try { await api("/api/runs", "POST", { target: bench, model_spec: model }); render(); }
  catch (e) { alert(e.message); }
}

function sampleResult(r) {
  const s = r.frontier_split;
  const list = r.frontier.map((t) => `<li>${esc(t)}</li>`).join("");
  return `<div class="panel"><strong>Sample — ${esc(r.student)} on ${esc(r.benchmark)}</strong>
    <p class="muted">teacher ${esc(r.teacher)} · ${r.variants_created.length} variant(s) created ·
      frontier ${r.frontier.length} (${s.learnability.length} learnable, ${s.only_incumbent.length} only-incumbent)</p>
    <ul>${list || `<li class="muted">frontier is empty — the student matches the incumbent</li>`}</ul></div>`;
}

function distillResult(r) {
  return `<div class="panel"><strong>Distill — ${esc(r.student)} on ${esc(r.benchmark)}</strong>
    <p class="muted">frontier ${r.frontier.length} · teacher demos ${r.demos.length} · backend ${esc(r.backend)} (${esc(r.status)})</p>
    ${r.out_dir ? `<p>datasets in <code>${esc(r.out_dir)}</code></p>` : ""}
    ${r.escalated && r.escalated.length ? `<p class="muted">escalated to rooms: ${r.escalated.map(esc).join(", ")}</p>` : ""}
    <p>${esc(r.plan)}</p></div>`;
}

async function createBenchmark() {
  const msg = document.getElementById("bm-msg");
  const tags = document.getElementById("bm-tags").value.split(",").map((s) => s.trim()).filter(Boolean);
  const body = { name: document.getElementById("bm-name").value };
  if (tags.length) body.tags = tags; else body.all = true;
  try { await api("/api/benchmarks", "POST", body); render(); }
  catch (e) { msg.textContent = e.message; }
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
    `<option value="${esc(b.name)}">${esc(b.name)} (${b.task_count} tasks)</option>`).join("");
  view.innerHTML = `<h2>Train</h2>
    <p class="muted">Prepare SFT / preference / RL datasets from a benchmark, then emit a runnable backend config. Nothing trains here — Touchstone hands you the datasets and the exact command to run on GPU infra.</p>
    <div class="form">
      <label>benchmark <select id="tr-bench">${opts}</select></label>
      <button id="tr-prepare">Prepare datasets</button>
    </div>
    <div id="tr-loop"></div>
    <div id="tr-out"></div>`;
  document.getElementById("tr-prepare").onclick = trainPrepare;
  document.getElementById("tr-bench").onchange = showLoopState;
  showLoopState();
}

async function showLoopState() {
  const bench = document.getElementById("tr-bench").value;
  const box = document.getElementById("tr-loop");
  try {
    const s = await api(`/api/loop/${encodeURIComponent(bench)}`);
    if (!s || (s.last_sample === undefined && s.last_distill === undefined)) {
      box.innerHTML = `<p class="muted">No Sample/Distill loop run yet for <code>${esc(bench)}</code> — start one on the Benchmarks page.</p>`;
      return;
    }
    const stamp = (t) => (t ? esc(String(t).slice(0, 19).replace("T", " ")) : "—");
    box.innerHTML = `<div class="panel">
      <strong>Loop</strong>
      <p class="muted">frontier ${esc(s.frontier_size ?? "—")} task(s) · student ${esc(s.student || "—")}</p>
      <p class="muted">last Sample: ${stamp(s.last_sample)} · last Distill: ${stamp(s.last_distill)}</p></div>`;
  } catch (e) { box.innerHTML = ""; }
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
      esc(r.task), r.incumbent_passed ? "pass" : "fail", r.candidate_passed ? "pass" : "fail", verdict[r.category],
    ]);
    const c = p.counts;
    out.innerHTML = `<p class="muted">both pass ${c.both_pass} · only incumbent ${c.only_incumbent} · only candidate ${c.only_candidate} · both fail ${c.both_fail}</p>` +
      table(["task", "incumbent", "candidate", "verdict"], rows);
  } catch (e) { out.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
}
