"use strict";

// Web Audio voice for the review room: realtime PCM16 capture/playback over the room WebSocket,
// and the local MediaRecorder push-to-talk. Loaded before room.js and shares its globals
// ($, wsSend, setVoiceState, notice, speaker, mode, ROOM_ID).

let captureCtx = null, procNode = null, micStream = null;
let playCtx = null, playHead = 0;
let playing = false, playTail = null;   // echo guard: never capture while the agent is speaking
let micOn = false;                       // realtime mic — always-on with server VAD unless muted
let recorder = null, chunks = [];        // local push-to-talk

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

// -- agent audio playback (with the echo guard) ------------------------------

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
  playing = true;
  clearTimeout(playTail);
  setVoiceState("speaking");
  node.onended = endAgentAudio;
}

function endAgentAudio() {
  if (!playCtx || playCtx.currentTime < playHead - 0.05) return;  // more audio still queued
  clearTimeout(playTail);
  playTail = setTimeout(() => {   // ~300ms tail so the mic doesn't catch the speaker's decay
    playing = false;
    setVoiceState(micOn ? "listening" : "mic off");
  }, 300);
}

// -- realtime mic (always-on, server VAD) ------------------------------------

async function openRealtimeMic() {
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
    if (!micOn || playing) return;   // muted, or the agent is speaking (echo guard)
    const pcm = floatToPCM16(e.inputBuffer.getChannelData(0));
    wsSend({ type: "audio", b64: bytesToBase64(new Uint8Array(pcm.buffer)), speaker });
  };
  src.connect(procNode);
  procNode.connect(captureCtx.destination);
  micOn = true;
  wsSend({ type: "ptt", state: "down", speaker });   // claim the shared floor once
  setVoiceState("listening");
  refreshTalkButton();
  return true;
}

function setMic(on) {
  micOn = on;
  wsSend({ type: "ptt", state: on ? "down" : "up", speaker });
  setVoiceState(on ? (playing ? "speaking" : "listening") : "mic off");
  refreshTalkButton();
}

// -- #talk button (round mic; the glyph is fixed markup, so we toggle state, not text) --------

function refreshTalkButton() {
  const btn = $("talk");
  if (!btn) return;
  const active = mode === "realtime" ? micOn : !!(recorder && recorder.state === "recording");
  btn.setAttribute("aria-pressed", active ? "true" : "false");
}

async function toggleTalk() {
  if (mode === "realtime") {
    if (!micOn && !procNode) { await openRealtimeMic(); refreshTalkButton(); return; }
    setMic(!micOn);
    return;
  }
  await toggleLocalTalk();
}

// -- local push-to-talk (MediaRecorder -> POST /audio) -----------------------

async function toggleLocalTalk() {
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
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
  recorder.onstop = () => finishLocalClip(stream);
  recorder.start();
  refreshTalkButton();
}

async function finishLocalClip(stream) {
  stream.getTracks().forEach((t) => t.stop());
  refreshTalkButton();
  const form = new FormData();
  form.append("speaker", speaker);
  form.append("file", new Blob(chunks, { type: "audio/webm" }), "clip.webm");
  const r = await fetch(`/api/rooms/${ROOM_ID}/audio`, { method: "POST", body: form });
  if (!r.ok) notice((await r.json().catch(() => ({}))).detail || "Audio failed.");
}

// -- local TTS playback ------------------------------------------------------

function speak(msg) {
  fetch(`/api/rooms/${ROOM_ID}/audio/${msg.id}`).then((r) => {
    if (r.status === 204) {
      if (window.speechSynthesis) speechSynthesis.speak(new SpeechSynthesisUtterance(msg.text));
      return;
    }
    if (r.ok) r.blob().then((b) => new Audio(URL.createObjectURL(b)).play().catch(() => {}));
  });
}
