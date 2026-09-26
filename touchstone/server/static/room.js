"use strict";

const ROOM_ID = location.pathname.split("/").pop();
const $ = (id) => document.getElementById(id);
let speaker = localStorage.getItem("touchstone.name") || "";
let lastAgentAudioId = null;
let mode = "local";
let socket = null;

function esc(s) {
  return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

let noticeTimer = null;
function notice(text) {
  const el = $("notice");
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("hidden", !text);
  clearTimeout(noticeTimer);
  if (text) noticeTimer = setTimeout(() => notice(""), 6000);
}

// -- conversation ------------------------------------------------------------

const seenCaptions = new Set();  // ids already on screen, so only genuinely new captions fade in

function renderMessages(messages) {
  const box = $("messages");
  box.innerHTML = messages
    .map((m) => {
      const fresh = m.id && !seenCaptions.has(m.id) ? " isNew" : "";
      const reviewer = m.role === "assistant";
      const who = reviewer ? "Reviewer" : (m.speaker || "you");
      return `<div class="turn ${reviewer ? "reviewer" : "person"}${fresh}">` +
        `<div class="who">${esc(who)}</div><div class="cap">${esc(m.text)}</div></div>`;
    })
    .join("");
  messages.forEach((m) => m.id && seenCaptions.add(m.id));
  box.scrollTop = box.scrollHeight;
  const lastAgent = [...messages].reverse().find((m) => m.role === "assistant");
  if (lastAgent && lastAgent.id !== lastAgentAudioId) {
    lastAgentAudioId = lastAgent.id;
    if (!micOn) setVoiceState("mic off");         // the reply arrived; drop the "thinking" word
    if (mode !== "realtime") speak(lastAgent);    // realtime voices replies over the WebSocket
  }
}

// -- review panels -----------------------------------------------------------

function pct(x) {
  return x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`;
}

function renderTrust(trust) {
  const el = $("trust");
  if (!trust || !trust.reviewed) { el.textContent = "trust — · no reviews yet"; return; }
  el.textContent = `trust ${pct(trust.score)} · ${trust.agreed} of ${trust.reviewed} reviewed`;
}

const QUEUE = [["unsure", "unsure"], ["disagree", "disagree"], ["unreviewed", "unreviewed"]];

function renderQueue(review) {
  const counts = (review && review.counts) || {};
  const parts = QUEUE
    .map(([key, label]) => (counts[key] ? `${counts[key]} ${label}` : ""))
    .filter(Boolean);
  const needs = (review && review.needs_review) || 0;
  if (needs) parts.push(`${needs} need review`);
  const el = $("queue");
  el.textContent = parts.join(" · ");
  const stale = counts.stale || 0;
  el.title = stale ? `${stale} trial(s) from renamed/removed tasks are hidden` : "";
}

function criterionRow(c) {
  const ok = c.score === 1 || c.score === true;
  const raw = c.raw && c.raw !== c.description ? ` title="${esc(c.raw)}"` : "";
  return `<li class="crit ${ok ? "pass" : "fail"}"><span class="dot" aria-hidden="true"></span>` +
    `<span class="desc"${raw}>${esc(c.description)}</span>` +
    `<span class="mark">${pct(c.score)}</span></li>`;
}

function renderProposed(review) {
  const el = $("trialProposed");
  if (!review || !review.proposed) { el.classList.add("hidden"); el.innerHTML = ""; return; }
  el.classList.remove("hidden");
  el.innerHTML = `<strong>Proposed change</strong><div>${esc(review.proposed.readback)}</div>` +
    `<div class="hint">Say “yes” to apply, or tell me what to change.</div>`;
}

function renderTrial(review) {
  const cur = review && review.current;
  $("trialEmpty").classList.toggle("hidden", !!cur);
  $("trialCard").classList.toggle("hidden", !cur);
  if (!cur) return;
  $("trialTask").textContent = cur.task;
  const width = Math.round(Math.max(0, Math.min(1, cur.reward || 0)) * 100);
  $("trialReward").innerHTML = `<span class="num">${pct(cur.reward)}</span>` +
    `<span class="bar"><span style="width:${width}%"></span></span>` +
    `<span class="cap">verifier reward</span>`;
  $("trialInstruction").textContent = cur.instruction || "—";
  $("trialTrajectory").innerHTML = (cur.trajectory || [])
    .map((line) => `<li class="step">${esc(line)}</li>`).join("") || "<li class='muted'>—</li>";
  $("trialCriteria").innerHTML = (cur.criteria || []).map(criterionRow).join("") ||
    "<li class='muted'>no criteria</li>";
  renderProposed(review);
}

function applyReview(review) {
  reviewState = review || {};
  renderTrust(reviewState.trust);
  renderQueue(reviewState);
  renderTrial(reviewState);
}

// The goal sentence for the top bar: the jobs the agent handles, lifted from the reviewer's
// opening line ("Your agent handles X, Y and Z; …"); falls back to the room topic.
function goalLine(messages, topic) {
  const opening = (messages || []).find((m) => m.role === "assistant");
  const match = opening && /handles (.+?)[;.]/.exec(opening.text || "");
  return match ? match[1] : topic;
}

let reviewState = {};
let messagesState = [];
let pollTimer = null;

function applyState(s) {
  $("topic").textContent = goalLine(s.messages, s.room.topic);
  $("closed").classList.toggle("hidden", !s.room.closed_at);
  if (!speaker && s.you) speaker = s.you;  // no join card: the server names the local user
  if (s.mode) { mode = s.mode; refreshTalkButton(); }
  messagesState = s.messages;
  renderMessages(messagesState);
  applyReview(s.review);
}

async function refresh() {
  const r = await fetch(`/api/rooms/${ROOM_ID}`);
  if (r.ok) applyState(await r.json());
}

function startPolling() {
  if (!pollTimer) pollTimer = setInterval(refresh, 1000);
}
function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/rooms/${ROOM_ID}`);
  socket = ws;
  // On (re)connect, re-render from authoritative state so a gap while disconnected can't leave
  // the DOM stale; the server also pushes a full "state" event at the end of every turn.
  ws.onopen = () => {
    stopPolling();
    if (!(mode === "realtime" && micOn)) setVoiceState("mic off");
    refresh();
  };
  ws.onmessage = (ev) => {
    const { type, data } = JSON.parse(ev.data);
    if (type === "state") applyState(data);
    else if (type === "review") applyReview(data);
    else if (type === "closed") { $("closed").classList.remove("hidden"); }
    else if (type === "audio") { playPCM(base64ToInt16(data.b64)); }
    else if (type === "message") { messagesState = messagesState.concat(data); renderMessages(messagesState); }
    else if (type === "status") { if (!micOn) setVoiceState(data && data.state ? data.state : "thinking"); }
    else if (type === "fallback") { mode = "local"; refreshTalkButton(); setVoiceState("local voice"); }
  };
  ws.onclose = () => {
    socket = null;
    startPolling();
    if (mode === "realtime") setVoiceState("reconnecting…");
    setTimeout(connect, 1500);
  };
}

function wsSend(obj) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(obj));
}

function setVoiceState(text) {
  const el = $("voiceState");
  if (el) { el.textContent = text; el.classList.toggle("hidden", !text); }
}

async function send(text) {
  if (!text.trim()) return;
  $("text").value = "";
  if (!micOn) setVoiceState("thinking");   // the reviewer is composing a reply (cleared on arrival)
  await fetch(`/api/rooms/${ROOM_ID}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speaker, text }),
  });
}

// -- start overlay -----------------------------------------------------------

function startedKey() { return `touchstone.started.${ROOM_ID}`; }
function sessionFlag(key) { try { return sessionStorage.getItem(key); } catch { return null; } }
function setSessionFlag(key, v) { try { sessionStorage.setItem(key, v); } catch { /* private mode */ } }

function showOverlay() {
  const started = sessionFlag(startedKey()) === "1";
  $("startBtn").textContent = started ? "▶ resume" : "▶ start";
  $("startHint").textContent = mode === "realtime"
    ? "Your browser will ask for the microphone once."
    : "Voice replies play in the browser; type or push to talk.";
  $("startOverlay").classList.remove("hidden");
}

async function startReview() {
  const resuming = sessionFlag(startedKey()) === "1";
  setSessionFlag(startedKey(), "1");
  $("startOverlay").classList.add("hidden");
  if (mode === "realtime") await openRealtimeMic();  // one gesture -> one mic permission
  refreshTalkButton();
  if (!resuming) send("start");  // opens the first trial via the normal /messages path
}

// -- boot -------------------------------------------------------------------

async function boot() {
  $("text").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); send($("text").value); }
  });
  $("talk").onclick = toggleTalk;
  $("next").onclick = () => send("Next trial, please.");
  $("startBtn").onclick = startReview;
  await refresh();   // learn the mode + speaker from room state before showing the overlay
  showOverlay();
  connect();
}

boot();
