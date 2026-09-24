"use strict";

const ROOM_ID = location.pathname.split("/").pop();
const $ = (id) => document.getElementById(id);
let speaker = localStorage.getItem("touchstone.name") || "";
let lastAgentAudioId = null;

function esc(s) {
  return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function renderMessages(messages) {
  const box = $("messages");
  box.innerHTML = messages
    .map((m) => {
      const cls = m.role === "assistant" ? "msg agent" : "msg";
      const who = m.role === "assistant" ? "Interviewer" : m.speaker;
      return `<div class="${cls}"><div class="who"><b>${esc(who)}</b></div>` +
        `<div class="body">${esc(m.text)}</div></div>`;
    })
    .join("");
  box.scrollTop = box.scrollHeight;
  const lastAgent = [...messages].reverse().find((m) => m.role === "assistant");
  if (lastAgent && lastAgent.id !== lastAgentAudioId) {
    lastAgentAudioId = lastAgent.id;
    speak(lastAgent);
  }
}

function checkCard(c, committed) {
  const params = JSON.stringify(c.params);
  const commitBtn = committed
    ? `<span class="badge">${esc(c.severity)} · committed</span>`
    : `<button data-check="${c.id}" class="commitBtn">Commit</button>`;
  return `<div class="check ${committed ? "committed" : ""}">` +
    `<div class="kind">${esc(c.kind)}</div>` +
    `<div class="name">${esc(c.name || c.kind)}</div>` +
    `<div class="params">${esc(params)}</div>` +
    (c.rationale ? `<div class="rationale">${esc(c.rationale)}</div>` : "") +
    commitBtn + `</div>`;
}

function renderChecks(draft, committed) {
  $("draftList").innerHTML = draft.length
    ? draft.map((c) => checkCard(c, false)).join("")
    : `<div class="badge">nothing drafted yet</div>`;
  $("committedList").innerHTML = committed.map((c) => checkCard(c, true)).join("");
  document.querySelectorAll(".commitBtn").forEach((b) => {
    b.onclick = () => send("/commit");
  });
}

let draftState = [];
let committedState = [];

function applyState(s) {
  $("topic").textContent = s.room.topic;
  $("closed").classList.toggle("hidden", !s.room.closed_at);
  draftState = s.draft;
  committedState = s.committed;
  renderMessages(s.messages);
  renderChecks(draftState, committedState);
}

async function refresh() {
  const r = await fetch(`/api/rooms/${ROOM_ID}`);
  if (r.ok) applyState(await r.json());
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/rooms/${ROOM_ID}`);
  ws.onmessage = (ev) => {
    const { type, data } = JSON.parse(ev.data);
    if (type === "state") applyState(data);
    else if (type === "draft") { draftState = data.checks; renderChecks(draftState, committedState); }
    else if (type === "committed") refresh();
    else if (type === "closed") { $("closed").classList.remove("hidden"); }
    else if (type === "message") refresh();
  };
  ws.onclose = () => setTimeout(connect, 1500);
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

// -- push to talk -----------------------------------------------------------

let recorder = null;
let chunks = [];

async function toggleTalk() {
  const btn = $("talk");
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    alert("Microphone not available.");
    return;
  }
  recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    btn.classList.remove("recording");
    const blob = new Blob(chunks, { type: "audio/webm" });
    const form = new FormData();
    form.append("speaker", speaker);
    form.append("file", blob, "clip.webm");
    const r = await fetch(`/api/rooms/${ROOM_ID}/audio`, { method: "POST", body: form });
    if (!r.ok) alert((await r.json().catch(() => ({}))).detail || "Audio failed.");
  };
  recorder.start();
  btn.classList.add("recording");
}

// -- boot -------------------------------------------------------------------

function boot() {
  $("send").onclick = () => send($("text").value);
  $("text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("text").value); }
  });
  $("talk").onclick = toggleTalk;
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
