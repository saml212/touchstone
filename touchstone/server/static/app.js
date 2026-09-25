"use strict";

// The five-page shell over the dataset + Harbor job dirs. Every page reads a /api/pages/* route at
// request time; the Review page hands off to the stage-4 room at /rooms/{id}. Vanilla JS, no build.

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

function link(hash, text) {
  return `<a href="#${hash}">${esc(text)}</a>`;
}

function table(headers, rows) {
  const head = headers.map((h) => `<th>${esc(h)}</th>`).join("");
  const body = rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function pct(reward) {
  return reward == null ? "—" : `${Math.round(reward * 100)}%`;
}

function fmtJob(job) {
  const [date, time] = String(job).split("__");
  return time ? `${date} ${time.replace(/-/g, ":")}` : job;
}

// ---- router ----------------------------------------------------------------

const PAGES = {
  overview: overviewPage,
  tasks: tasksPage,
  trials: trialsPage,
  trial: trialDetail,
  review: reviewPage,
  train: trainPage,
};

async function render() {
  const parts = (location.hash.replace(/^#\/?/, "") || "overview").split("/").map(decodeURIComponent);
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

// ---- Overview --------------------------------------------------------------

async function overviewPage() {
  const o = await api("/api/pages/overview");
  if (!o.has_dataset) {
    view.innerHTML = `<h2>Overview</h2><p class="muted">No dataset here yet — run
      <code>touchstone survey</code> to build <code>touchstone/</code>, then reload.</p>`;
    return;
  }
  view.innerHTML = `
    <div class="lede">${esc(o.sentence)}</div>
    <a class="cta" href="#/review">Walk through the trials →</a>
    ${o.map ? `<p class="map">${esc(o.map)}</p>` : ""}
    ${overviewJobs(o.jobs_to_be_done)}
    ${overviewServices(o.services)}
    ${overviewLatest(o.latest_jobs)}
    ${overviewTrust(o.trust)}`;
}

function overviewJobs(jobs) {
  const entries = Object.entries(jobs || {});
  if (!entries.length) return "";
  const rows = entries.map(([job, n]) => [esc(job), esc(n)]);
  return `<h3>Jobs to be done</h3>${table(["job", "tasks"], rows)}`;
}

function overviewServices(services) {
  if (!services || !services.length) return "";
  const rows = services.map((s) => [
    esc(s.service), pct(s.fidelity),
    `${esc(s.reproduced ?? "?")}/${esc(s.calls ?? "?")}`,
    s.status === "ok" ? badge("ok") : badge("flagged", "outcome"),
  ]);
  return `<h3>Services &amp; fidelity</h3>${table(["service", "fidelity", "reproduced", "status"], rows)}`;
}

function overviewLatest(latest) {
  if (!latest || !latest.length) return "";
  const rows = latest.map((j) => [
    link(`/trials/${encodeURIComponent(j.job)}`, fmtJob(j.job)),
    esc(j.model || j.agent), `${esc(j.passed)}/${esc(j.tasks)}`,
  ]);
  return `<h3>Latest job per model</h3>${table(["job", "model", "passed"], rows)}`;
}

function overviewTrust(trust) {
  if (!trust) return "";
  return `<div class="hint">Trust: the human agreed with the verifier on
    <b>${trust.agreed}/${trust.reviewed}</b> reviewed trials (${pct(trust.score)}).</div>`;
}

function badge(label, cls = "") {
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}

// ---- Tasks -----------------------------------------------------------------

async function tasksPage(args) {
  if (args[0]) return taskDetail(args[0]);
  const { tasks } = await api("/api/pages/tasks");
  if (!tasks.length) {
    view.innerHTML = `<h2>Tasks</h2><p class="muted">No tasks yet — run <code>touchstone survey</code>.</p>`;
    return;
  }
  const rows = tasks.map((t) => [
    link(`/tasks/${encodeURIComponent(t.task)}`, t.task),
    esc(t.job || "—"),
    critList(t.criteria),
    gateCell(t.gate),
    rewardsCell(t.rewards),
  ]);
  view.innerHTML = `<h2>Tasks <span class="count">${tasks.length}</span></h2>
    ${table(["task", "job to be done", "what the verifier checks", "gate", "last reward"], rows)}`;
}

function critList(criteria) {
  if (!criteria || !criteria.length) return `<span class="muted">—</span>`;
  return `<ul class="mini">${criteria.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>`;
}

function gateCell(gate) {
  if (!gate || (gate.oracle == null && gate.nop == null)) return `<span class="muted">—</span>`;
  return `<code>oracle ${pct(gate.oracle)} · nop ${pct(gate.nop)}</code>`;
}

function rewardsCell(rewards) {
  const entries = Object.entries(rewards || {});
  if (!entries.length) return `<span class="muted">—</span>`;
  return entries.map(([model, r]) => `<div><span class="muted">${esc(model)}</span> ${pct(r)}</div>`).join("");
}

async function taskDetail(name) {
  const t = await api(`/api/pages/tasks/${encodeURIComponent(name)}`);
  view.innerHTML = `<div class="crumb">${link("/tasks", "← Tasks")}</div>
    <h2>${esc(t.task)} ${t.job ? badge(t.job) : ""}</h2>
    <h3>What the customer asked</h3>
    <p class="instruction">${esc(t.instruction) || `<span class="muted">—</span>`}</p>
    ${t.persona ? `<h3>Persona</h3><p class="instruction">${esc(t.persona)}</p>` : ""}
    <h3>Criteria &amp; weights</h3>${criteriaByDim(t.criteria, t.weights)}
    <h3>Trials <span class="count">${t.trials.length}</span></h3>${taskTrials(name, t.trials)}`;
}

function criteriaByDim(criteria, weights) {
  const dims = Object.entries(criteria || {});
  if (!dims.length) return `<p class="muted">No criteria.</p>`;
  return dims.map(([dim, list]) => {
    const w = (weights || {})[dim];
    const wtag = w == null ? "" : `<span class="badge">weight ${esc(w)}</span>`;
    const items = list.map((c) => `<li>${esc(c)}</li>`).join("");
    return `<div class="dimblock"><div class="dimhead"><code>${esc(dim)}</code> ${wtag}</div>
      <ul class="mini">${items}</ul></div>`;
  }).join("");
}

function taskTrials(name, trials) {
  if (!trials.length) return `<p class="muted">No trials recorded yet.</p>`;
  const rows = trials.map((t) => [
    trialLink(name, t.trial, fmtJob(t.job)),
    esc(t.model || "—"), pct(t.reward),
  ]);
  return table(["job", "model", "reward"], rows);
}

function trialLink(task, trialId, text) {
  const [job, dir] = String(trialId).split("/");
  const hash = `/trial/${encodeURIComponent(task)}/${encodeURIComponent(job)}/${encodeURIComponent(dir)}`;
  return link(hash, text);
}

// ---- Trials (job picker + one trial) ---------------------------------------

async function trialsPage(args) {
  if (args[0]) return jobRewards(args[0]);
  const { jobs } = await api("/api/pages/jobs");
  if (!jobs.length) {
    view.innerHTML = `<h2>Trials</h2><p class="muted">No jobs yet — run <code>touchstone bench</code>.</p>`;
    return;
  }
  const rows = jobs.map((j) => [
    link(`/trials/${encodeURIComponent(j.job)}`, fmtJob(j.job)),
    esc(j.model || j.agent), `${esc(j.passed)}/${esc(j.tasks)}`,
    j.gate ? badge("gate") : "",
  ]);
  view.innerHTML = `<h2>Trials <span class="count">${jobs.length}</span></h2>
    <p class="muted">Pick a job to see its per-task rewards.</p>
    ${table(["job", "agent / model", "passed", ""], rows)}`;
}

async function jobRewards(job) {
  const d = await api(`/api/pages/jobs/${encodeURIComponent(job)}`);
  const rows = d.rewards.map((r) => [
    trialLink(r.task, r.trial, r.task),
    esc(r.model || "—"), pct(r.reward),
  ]);
  view.innerHTML = `<div class="crumb">${link("/trials", "← Trials")}</div>
    <h2>${esc(fmtJob(d.job))} ${badge(d.model || d.agent)}</h2>
    ${rows.length ? table(["task", "model", "reward"], rows) : `<p class="muted">No trials.</p>`}`;
}

async function trialDetail(args) {
  const [task, job, dir] = args;
  const trialId = `${job}/${dir}`;
  const d = await api(`/api/pages/trial?task=${encodeURIComponent(task)}&trial=${encodeURIComponent(trialId)}`);
  view.innerHTML = `<div class="crumb">${link(`/tasks/${encodeURIComponent(task)}`, "← Task")}</div>
    <h2>${esc(d.task)}</h2>
    <div class="reward">reward ${pct(d.reward)} · ${esc(fmtJob(job))}</div>
    <button id="reviewBtn" data-task="${esc(d.task)}">Review this trial</button>
    <h3>What the customer asked</h3>
    <p class="instruction">${esc(d.instruction) || `<span class="muted">—</span>`}</p>
    <h3>What the agent did</h3>${transcript(d.trajectory)}
    <h3>What the verifier checked</h3>${criteriaScores(d.criteria)}`;
  document.getElementById("reviewBtn").onclick = () => startReview(d.task);
}

function transcript(steps) {
  if (!steps || !steps.length) return `<p class="muted">The agent did nothing.</p>`;
  return `<div class="trajectory">${steps.map((s) => `<div class="step">${esc(s)}</div>`).join("")}</div>`;
}

function criteriaScores(criteria) {
  if (!criteria || !criteria.length) return `<p class="muted">No criteria.</p>`;
  const items = criteria.map((c) => {
    const passed = c.score != null && c.score >= 1;
    const cls = c.score == null ? "" : passed ? "pass" : "fail";
    const mark = c.score == null ? "" : `<span class="mark">${pct(c.score)}</span>`;
    return `<li class="crit ${cls}"><span class="dim">${esc(c.dimension)}</span>
      <span class="desc" title="${esc(c.raw || "")}">${esc(c.description)}</span>${mark}</li>`;
  }).join("");
  return `<ul class="criteria">${items}</ul>`;
}

async function startReview(task) {
  const btn = document.getElementById("reviewBtn");
  btn.disabled = true;
  btn.textContent = "opening…";
  try {
    const state = await api("/api/rooms", "POST", { task_id: task, topic: `review of ${task}` });
    location.href = `/rooms/${state.room.id}`;
  } catch (e) {
    btn.disabled = false;
    btn.textContent = "Review this trial";
    alert(e.message);
  }
}

// ---- Review (link to the stage-4 room) -------------------------------------

async function reviewPage() {
  const { rooms } = await api("/api/rooms");
  const open = rooms.filter((r) => !r.closed_at);
  const rows = rooms.map((r) => [
    `<a href="/rooms/${esc(r.id)}">${esc(r.topic)}</a>`,
    esc(r.task_name || "—"),
    r.closed_at ? badge("closed") : badge("open", "outcome"),
  ]);
  view.innerHTML = `<h2>Review</h2>
    <p class="muted">A voice/text room over one task and its trials — agree with the verifier, or
      correct it and Harbor regrades. ${open.length} open.</p>
    <button id="startRoom">Start a review</button>
    <h3>Rooms</h3>
    ${rows.length ? table(["room", "task", "state"], rows) : `<p class="muted">No rooms yet.</p>`}`;
  document.getElementById("startRoom").onclick = async () => {
    const state = await api("/api/rooms", "POST", { topic: "review" });
    location.href = `/rooms/${state.room.id}`;
  };
}

// ---- Train -----------------------------------------------------------------

async function trainPage() {
  const t = await api("/api/pages/train");
  if (!t.present) {
    view.innerHTML = `<h2>Train</h2><p class="muted">No training data yet. Run
      <code>touchstone train</code> to route finished jobs into distill / RL / hold-out.</p>`;
    return;
  }
  const c = t.counts;
  const tiles = [
    ["Distill", c.distill], ["RL", c.rl], ["Hold-out", c.hold_out], ["Stuck", c.stuck],
  ].map(([k, v]) => `<div class="tile"><div class="num">${esc(v)}</div><div class="lbl">${esc(k)}</div></div>`).join("");
  view.innerHTML = `<h2>Train</h2>
    <p class="muted">Each task routes by its pass rate: 0 → distill · 0–1 → RL · 1 → hold out.
      ${t.trajectories} distill trajectories · threshold ${esc(t.threshold)}.</p>
    <div class="tiles">${tiles}</div>
    <h3>Re-run</h3>
    <pre id="cmd">${esc(t.command)}</pre>
    <button id="copyCmd" class="ghost">Copy command</button>
    <h3>Inputs</h3>${trainInputs(t.inputs)}`;
  wireCopy(t.command);
}

function trainInputs(inputs) {
  const rows = [];
  for (const role of ["teacher", "student"]) {
    for (const job of inputs[role] || []) {
      rows.push([esc(role), esc(fmtJob(job.dir)), esc(job.model || job.agent || "—")]);
    }
  }
  return rows.length ? table(["role", "job", "model"], rows) : `<p class="muted">—</p>`;
}

function wireCopy(command) {
  document.getElementById("copyCmd").onclick = async () => {
    try {
      await navigator.clipboard.writeText(command);
      document.getElementById("copyCmd").textContent = "copied ✓";
    } catch (e) {
      document.getElementById("copyCmd").textContent = "copy failed";
    }
  };
}
