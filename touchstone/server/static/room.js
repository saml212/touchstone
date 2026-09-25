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

function talkLabel() {
  return mode === "realtime" ? "● hold to talk" : "● talk";
}

function setTalking(holding) {
  $("talk").textContent = holding ? `● ${speaker} holding` : talkLabel();
}

// -- conversation ------------------------------------------------------------

function renderMessages(messages) {
  const box = $("messages");
  box.innerHTML = messages
    .map((m) => {
      const cls = m.role === "assistant" ? "msg agent" : "msg";
      const who = m.role === "assistant" ? "Reviewer" : m.speaker;
      return `<div class="${cls}"><div class="who"><b>${esc(who)}</b></div>` +
        `<div class="body">${esc(m.text)}</div></div>`;
    })
    .join("");
  box.scrollTop = box.scrollHeight;
  const lastAgent = [...messages].reverse().find((m) => m.role === "assistant");
  if (lastAgent && lastAgent.id !== lastAgentAudioId) {
    lastAgentAudioId = lastAgent.id;
    if (mode !== "realtime") speak(lastAgent);
  }
}

// -- review panels -----------------------------------------------------------

function pct(x) {
  return x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`;
}

function renderTrust(trust) {
  const el = $("trust");
  if (!trust || !trust.reviewed) { el.textContent = "trust — (no reviews yet)"; return; }
  el.textContent = `trust ${pct(trust.score)} · agreed ${trust.agreed}/${trust.reviewed}`;
}

const FILTERS = [
  ["unsure", "unsure"], ["disagree", "disagree"],
  ["unreviewed", "unreviewed"], ["needs_review", "needs review"],
];

function renderChips(review) {
  const counts = (review && review.counts) || {};
  const parts = FILTERS.map(([key, label]) => {
    const n = key === "needs_review" ? (review && review.needs_review) || 0 : counts[key] || 0;
    return `<button class="chip" data-filter="${key}">${label} ${n}</button>`;
  });
  $("chips").innerHTML = parts.join("");
  document.querySelectorAll(".chip").forEach((b) => {
    b.onclick = () => send(`Show me the ${b.dataset.filter.replace("_", " ")} trials.`);
  });
}

function criterionRow(c) {
  const ok = c.score === 1 || c.score === true;
  const mark = c.score === null || c.score === undefined ? "" : ok ? "✓" : "✗";
  return `<li class="crit ${ok ? "pass" : "fail"}"><span class="dim">${esc(c.dimension)}</span>` +
    `<span class="desc">${esc(c.description)}</span><span class="mark">${mark} ${pct(c.score)}</span></li>`;
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
  $("trialReward").textContent = `verifier reward ${pct(cur.reward)}`;
  $("trialInstruction").textContent = cur.instruction || "—";
  $("trialTrajectory").innerHTML = (cur.trajectory || [])
    .map((line) => `<div class="step">${esc(line)}</div>`).join("") || "<div class='muted'>—</div>";
  $("trialCriteria").innerHTML = (cur.criteria || []).map(criterionRow).join("") ||
    "<li class='muted'>no criteria</li>";
  renderProposed(review);
}

function applyReview(review) {
  reviewState = review || {};
  renderTrust(reviewState.trust);
  renderChips(reviewState);
  renderTrial(reviewState);
}

let reviewState = {};
let messagesState = [];
let pollTimer = null;

function applyState(s) {
  $("topic").textContent = s.room.topic;
  $("closed").classList.toggle("hidden", !s.room.closed_at);
  if (s.mode) { mode = s.mode; setTalking(false); }
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
  ws.onopen = () => { stopPolling(); if (!realtimeTalking) setVoiceState(""); };
  ws.onmessage = (ev) => {
    const { type, data } = JSON.parse(ev.data);
    if (type === "state") applyState(data);
    else if (type === "review") applyReview(data);
    else if (type === "closed") { $("closed").classList.remove("hidden"); }
    else if (type === "audio") { playPCM(base64ToInt16(data.b64)); }
    else if (type === "message") { messagesState = messagesState.concat(data); renderMessages(messagesState); }
    else if (type === "fallback") { mode = "local"; setTalking(false); setVoiceState("● fell back to local voice"); }
  };
  ws.onclose = () => {
    socket = null;
    startPolling();
    if (mode === "realtime") setVoiceState("● reconnecting…");
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

// -- realtime voice (Web Audio PCM16 @ 24kHz over the WebSocket) -------------

let captureCtx = null, procNode = null, micStream = null;
let playCtx = null, playHead = 0;

function floatToPCM16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function bytesToBase64(bytes) {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function base64ToInt16(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

function playPCM(int16) {
  if (!playCtx) playCtx = new AudioContext({ sampleRate: 24000 });
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 0x8000;
  const buf = playCtx.createBuffer(1, f32.length, 24000);
  buf.getChannelData(0).set(f32);
  const node = playCtx.createBufferSource();
  node.buffer = buf;
  node.connect(playCtx.destination);
  playHead = Math.max(playHead, playCtx.currentTime);
  node.start(playHead);
  playHead += buf.duration;
  setVoiceState("● speaking");
  node.onended = () => { if (playCtx && playCtx.currentTime >= playHead - 0.05) setVoiceState(""); };
}

async function startRealtimeTalk() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    notice("Microphone not available — check your browser's mic permission.");
    return false;
  }
  captureCtx = new AudioContext({ sampleRate: 24000 });
  const src = captureCtx.createMediaStreamSource(micStream);
  procNode = captureCtx.createScriptProcessor(4096, 1, 1);
  procNode.onaudioprocess = (e) => {
    const pcm = floatToPCM16(e.inputBuffer.getChannelData(0));
    wsSend({ type: "audio", b64: bytesToBase64(new Uint8Array(pcm.buffer)), speaker });
  };
  src.connect(procNode);
  procNode.connect(captureCtx.destination);
  wsSend({ type: "ptt", state: "down", speaker });
  setVoiceState("● listening");
  return true;
}

function stopRealtimeTalk() {
  wsSend({ type: "ptt", state: "up", speaker });
  if (procNode) procNode.disconnect();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (captureCtx) captureCtx.close();
  procNode = micStream = captureCtx = null;
  setVoiceState("");
}

async function send(text) {
  if (!text.trim()) return;
  $("text").value = "";
  await fetch(`/api/rooms/${ROOM_ID}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speaker, text }),
  });
}

function speak(msg) {
  fetch(`/api/rooms/${ROOM_ID}/audio/${msg.id}`).then((r) => {
    if (r.status === 204) {
      if (window.speechSynthesis) speechSynthesis.speak(new SpeechSynthesisUtterance(msg.text));
      return;
    }
    if (r.ok) r.blob().then((b) => new Audio(URL.createObjectURL(b)).play().catch(() => {}));
  });
}

// -- local push-to-talk (MediaRecorder -> POST /audio) ----------------------

let recorder = null;
let chunks = [];
let realtimeTalking = false;

async function toggleTalk() {
  if (mode === "realtime") {
    const btn = $("talk");
    if (realtimeTalking) { realtimeTalking = false; btn.classList.remove("recording"); setTalking(false); stopRealtimeTalk(); }
    else if (await startRealtimeTalk()) { realtimeTalking = true; btn.classList.add("recording"); setTalking(true); }
    return;
  }
  const btn = $("talk");
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    notice("Microphone not available — check your browser's mic permission.");
    return;
  }
  recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    btn.classList.remove("recording");
    setTalking(false);
    const blob = new Blob(chunks, { type: "audio/webm" });
    const form = new FormData();
    form.append("speaker", speaker);
    form.append("file", blob, "clip.webm");
    const r = await fetch(`/api/rooms/${ROOM_ID}/audio`, { method: "POST", body: form });
    if (!r.ok) notice((await r.json().catch(() => ({}))).detail || "Audio failed.");
  };
  recorder.start();
  btn.classList.add("recording");
  setTalking(true);
}

// -- boot -------------------------------------------------------------------

function boot() {
  $("send").onclick = () => send($("text").value);
  $("text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("text").value); }
  });
  $("talk").onclick = toggleTalk;
  $("next").onclick = () => send("Next trial, please.");
  refresh();
  connect();
}

if (speaker) {
  $("join").classList.add("hidden");
  boot();
} else {
  $("joinBtn").onclick = () => {
    speaker = $("name").value.trim();
    if (!speaker) return;
    localStorage.setItem("touchstone.name", speaker);
    $("join").classList.add("hidden");
    boot();
  };
}
