# Touchstone — design

A touchstone is the black stone assayers rubbed gold against to prove its purity before anyone paid for it.
Touchstone does that for LLM agents: one line of code captures what your agent actually does, the traces
become a benchmark your stakeholders agree is fair, and the benchmark proves whether a cheaper or open model
is good enough **before** you switch or train.

Brand: Touchstone (product) by Pebble ML. Aesthetic: cream `#FAF5E7`, brick `#8B2E1F`, hairlines,
Space Grotesk + JetBrains Mono, light mode.

## Non-negotiables

- Everything runs locally on a laptop. No cloud service is required. No paid service is required.
- Zero-key demo: `touchstone demo` produces real traces, mined checks, a benchmark and a scoreboard using
  the built-in `scripted` provider. Real providers are additive.
- Python 3.12+, `uv`-managed, `pyproject.toml`, single package `touchstone`, one SQLite file. No ORM,
  no Postgres, no Docker for the core loop. Docker is needed only to run exported Harbor tasks.
- Every module has pytest coverage of edge cases, not just happy paths. `uv run pytest -q` must pass.
- Training is a clean hook (`touchstone.train.Trainer`), not an implementation. It must be obvious where
  GPU infra plugs in, and `touchstone train prepare` must produce the real datasets today.
- Small and readable beats clever. Prefer stdlib. Every dependency must earn its place.
- Secrets: env var first, then macOS Keychain (`security find-generic-password -a sam -s <name> -w`).
  Never print a secret. Never require one.

## The loop (what the user sees)

```
pip install touchstone            # 0. (from GitHub for now)
import touchstone; touchstone.trace()   # 1. one line: traces land in ./.touchstone/touchstone.db
touchstone serve                  # 2. local UI: episodes, checks, tasks, interview rooms, scoreboard
touchstone mine                   # 3. agent reads traces + code, proposes checks + tasks
touchstone interview <task>       # 4. voice/text room; stakeholders turn opinions into checks
touchstone bench run -m openai:gpt-4o-mini -m openai-compatible:http://localhost:8000/v1:qwen  # 5. proof
touchstone train prepare          # 6. SFT / preference / RL datasets ready for infra
touchstone export harbor          # 7. tasks as a Harbor benchmark (`harbor run` when Docker exists)
```

## Package layout

```
touchstone/
  __init__.py       trace(), episode(), outcome(), tool(), record_llm_call()  — the public one-liners
  config.py         Settings: db path, providers, speech, keychain prefix; from touchstone.toml + env
  store.py          SQLite (WAL) schema + typed dataclasses + all queries. The only place SQL lives.
  capture/          patch_openai.py, patch_anthropic.py, litellm.py (callback), context.py (episode ctx),
                    atif.py (Harbor ATIF export of an episode)
  llm/              Provider protocol `chat(messages, tools=None, json=False) -> Reply`;
                    scripted.py (deterministic, for tests/demo), reference.py (replays a task's
                    recorded reply — the incumbent baseline), claude_cli.py (`claude -p`, subscription),
                    codex_cli.py (`codex exec`), openai_compat.py (OpenAI + any /v1 endpoint), anthropic.py
                    spec strings: "scripted", "reference", "claude-cli:sonnet", "openai:gpt-4o-mini",
                    "anthropic:claude-sonnet-4-5", "openai-compatible:<base_url>:<model>", "codex-cli:gpt-5.6-sol".
                    The runner passes the task on its provider call path; only `reference` reads it,
                    every other provider keeps its unchanged signature and ignores it.
  checks/           dsl.py (Check dataclass + kinds), run.py (evaluate checks against an output), judge.py
  mine/             miner.py (statistical + LLM proposals of checks), cut.py (episodes -> replay tasks),
                    codebase.py (find system prompts / tool schemas in a repo)
  interview/        rooms.py (state), agent.py (question policy -> committed checks), speech.py (STT/TTS providers)
  bench/            benchmark.py (assemble), runner.py (replay tasks against candidates, async, retries),
                    report.py (scoreboard, proof table), harbor_export.py (Harbor task dirs), harbor_run.py
  train/            trainer.py (Protocol + NullTrainer), datasets.py (sft/preference/rl exports),
                    art.py, trl.py (adapters that write configs + commands; raise clearly when no infra)
  server/           app.py (FastAPI), routes/*.py, ws.py (rooms), static/ (index.html, app.js, app.css)
  cli.py            Typer: init, doctor, demo, serve, mine, checks, tasks, interview, bench, export, train
```

## Data model (store.py)

- `episodes(id, name, source, started_at, ended_at, outcome_score REAL, outcome_label, meta JSON)`
- `spans(id, episode_id, parent_id, kind ['llm','tool'], name, model, started_at, ended_at,
   input JSON, output JSON, tokens_in, tokens_out, cost_usd, error)`
  - llm input = `{messages:[...], tools:[...], params:{}}`; output = `{message:{role,content,tool_calls}}`
  - tool input = `{name, arguments}`; output = `{result}`
  - Messages everywhere in the store are the ONE canonical shape (`touchstone/messages.py`):
    `{role: system|user|assistant|tool, content: str, tool_calls?: [{id, name, arguments: str}],
    tool_call_id?: str, name?: str}`. `arguments` is always a JSON string. Capture points call
    `canonical()` (from OpenAI/Anthropic wire or already-canonical); the HTTP providers call
    `to_openai()` / `to_anthropic()` to convert back to wire shape when replaying.
- `checks(id, name, kind, params JSON, applies_to ['final','any_turn','tool_calls'], severity ['hard','soft'],
   source ['mined','interview','manual'], rationale, enabled INT, created_at)`
- `tasks(id, name, episode_id, cut_span_id, context JSON {messages, tools}, reference JSON, check_ids JSON,
   kind ['replay','harbor'], tags JSON, created_at)`
- `benchmarks(id, name, task_ids JSON, created_at)`
- `runs(id, benchmark_id, model_spec, started_at, finished_at, meta JSON)`
- `results(run_id, task_id, passed INT, check_results JSON, output JSON, latency_ms, cost_usd, error)`
- `rooms(id, task_id, topic, created_at, closed_at)`, `room_messages(id, room_id, speaker, role, text, audio_path, ts)`,
  `room_checks(room_id, check_id)`
IDs are `ulid`-style sortable strings generated in Python (no dependency). All timestamps ISO-8601 UTC.
SQLite opened with WAL + busy_timeout=5000; safe under concurrent writers (server + instrumented app).
Connections are opened `check_same_thread=False` and each caller (the CLI, each web request, each
background run thread) holds its own — created, used, and closed without being shared across threads
concurrently — so the threadpool and the instrumented app never collide on one connection.

## Checks DSL

`Check(kind, params)` evaluated against a `Target` = `{output_text, tool_calls:[{name,arguments}], messages, reference}`.
Kinds, programmatic first:
`contains`, `not_contains` (list, case-insensitive, any|all), `regex`, `not_regex`, `json_schema`,
`tool_called` (name, optional `arguments_match` dict of exact/regex), `tool_not_called`, `tool_order` (list),
`max_length`, `min_length`, `no_pii` (email/phone/card patterns), `expr` (safe expression via `simpleeval`
over `output`, `tools`, `reference`), `judge` (LLM rubric; soft by default; needs a provider; skipped and
reported as `skipped` when none).
Result: `CheckResult(check_id, passed: bool|None, evidence: str)`. A task passes when every enabled hard
check passes and no hard check errored. Soft checks are reported, never gate.
`regex`/`not_regex`/`tool_called.arguments_match` patterns are guarded against catastrophic
backtracking: a quantified group whose body itself repeats unboundedly (e.g. `(a+)+`, `([a-z]+)*`) is
rejected at validation and refused (errored result) at evaluation, so a pathological pattern can never
hang the engine.

## Mining

Input: episodes (optionally filtered), optional `--code <path>`. Steps:
1. Statistics (no LLM): tool usage per outcome label, output length distribution, JSON-ness, recurring
   phrases in good vs bad outcomes, PII leaks. Emits candidate checks with evidence counts.
2. LLM proposals (provider from config `agent_provider`, default `codex-cli`): sample of episodes +
   discovered system prompts and tool schemas → JSON list of checks in the DSL with rationale and
   supporting episode ids. Invalid proposals are dropped with a logged reason, never crash.
3. Cutting: every recorded assistant turn in every episode becomes a replay task (not only the final
   turn). Context = messages before the cut + tools; reference = recorded assistant message. A check
   attaches to a task only when the reference passes it (reference-consistent attachment), so a
   candidate is never asked to satisfy a check the recorded behaviour did not — except safety checks
   ("avoid this": `no_pii`, `not_contains`, `not_regex`, `tool_not_called`), which always attach.
   Tasks from episodes with bad outcomes get tag `failure` and only safety checks.
Everything mined is `enabled=0` until a human (or an interview) enables it. Idempotent: re-mining does not
duplicate identical checks/tasks.

## Interview

A room is attached to a task (or a failure set). Participants join `/rooms/<id>` from any browser with a
display name; messages broadcast over WebSocket. The agent (LLM) opens with a summary of the task and the
model's output, asks one concrete question at a time ("What should it have done instead?", "Is that a hard
rule or a preference?", "Would this wording pass?"), maintains a draft check list visible to everyone, and
commits a check when a participant confirms. The agent prefers a programmatic check kind that states
the rule exactly and reaches for `judge` only when no programmatic kind can express it. A bare "yes"
commits the current draft as-is; a confirmation that carries an amendment ("yes, and one L is fine
too") is revised through the LLM first and only then committed, so the stored check reflects the
amendment. Committed checks are `source='interview'`, `enabled=1`,
attached to the task. Multiplayer: no host; any participant can confirm; the agent addresses people by name
and reconciles disagreement by asking the group.
Voice: browser push-to-talk (MediaRecorder) → `POST /rooms/{id}/audio` → STT → message. Agent replies →
TTS → audio to all clients. Providers: STT `faster-whisper` (local, optional extra), `openai` (whisper-1),
`none`; TTS `openai`, `say` (macOS), `browser` (speechSynthesis, zero-dep default). Auto-detect in
`doctor`. Text-only always works.

## Bench

`Benchmark` = task ids. Runner replays each task's context to each candidate `model_spec`, records the
reply as output, evaluates checks, stores results. Async with a semaphore, per-call timeout, 2 retries
on transport errors, cost from token counts × a small price table (unknown model → null cost, never a guess).
Report: pass rate per model, per check, per tag; proof table "candidate vs incumbent"; JSON + terminal
table + UI. Determinism: `scripted` provider yields identical results on rerun. The `reference`
model spec is the honest incumbent: it replays each task's recorded reply, so `bench run -m reference`
passes every task whose attached checks the reference satisfies (all non-failure tasks by construction,
since a non-safety check attaches only when the reference already passes it) — the baseline a cheaper
candidate is proven against, and a way to confirm attached checks are satisfiable.

Harbor export: `touchstone export harbor --benchmark X --out ./harbor-tasks` writes one task dir per
task in the exact layout Harbor expects (read `~/Pebble/Github/harbor` to confirm: `instruction.md`,
`task.toml`, `environment/Dockerfile`, `tests/test.sh`, `tests/test_outputs.py`, `solution/solve.sh`).
The container test replays checks with a vendored `checks` module against `/logs/agent/output.json`.
`touchstone bench harbor-run` shells out to `harbor run` when Docker is available; otherwise prints the
exact command and where Docker is missing.

## Train

`Trainer` Protocol: `prepare(conn, benchmark_id, out_dir) -> DatasetBundle` and
`submit(bundle, config) -> JobHandle`. `datasets.py` writes, all atomically (re-runs overwrite),
under `.touchstone/train/<benchmark>/`: `sft.jsonl` (canonical context + reference completion for
tasks whose reference passes every attached hard check), `preference.jsonl` (per task, `{prompt,
chosen, rejected}` pairs — a passing candidate reply over a failing one from run results, plus the
reference over each failing candidate on tasks whose reference is good), `rl_tasks.jsonl` (task id,
context, tools, serialized attached checks for the reward verifier), and `manifest.json` (counts +
benchmark id). `NullTrainer.submit` writes `train_plan.md` and returns a handle with status
"planned". `art.py` writes `art_train.py` (an `art.TrainableModel` + a rollout that replays a task and
scores it with the vendored `touchstone_checks`, mirroring `bench/harbor_export`); `trl.py` writes
`trl_sft.yaml` + `run_trl.sh`. Both write their files and then raise `InfraRequired` with a one-line
instruction naming the file and command. `touchstone train prepare BENCH [--out DIR]` and
`train submit BENCH --backend null|art|trl [--out DIR]`; API `POST /api/train/prepare|submit` and a
download route for the jsonl files. Everything runs off the store; no GPU, no network.

## Server

FastAPI + uvicorn, single-page static UI (vanilla JS, no build step). Pages: Overview, Episodes (detail
with spans), Checks (toggle/edit), Tasks, Rooms (interview), Benchmarks (runs, scoreboard, proof table),
Train (prepare datasets, download the jsonl files, submit to a backend). JSON API under `/api/*`,
WebSocket `/ws/rooms/{id}`. The room page uses the WebSocket as the live channel and falls back to
polling `/api/rooms/{id}` only while the socket is down. Everything the CLI does the API does, and vice
versa (both call the same functions).

## CLI

`touchstone init` (writes `touchstone.toml`, `.touchstone/`; `--keychain-prefix` pins the Keychain
prefix, else the default `touchstone-` is left implicit), `doctor` (one table: python, db, SDKs,
providers, speech, `harbor`/`docker`/`ffmpeg` on PATH, and the `art`/`trl` train backends importable —
never installs, never a network call, always exits 0), `demo` (runs the built-in scripted support
agent: 30 episodes, mixed outcomes), `serve`,
`mine [--code PATH] [--provider SPEC]`, `checks list|enable|disable`, `tasks list|show`,
`interview <task_id>` (prints room URL, opens browser), `bench create|run|report`, `export harbor|atif`,
`train prepare|submit`. Exit codes non-zero on failure; errors are one clear sentence.

## Quality bar

- `uv run pytest -q` green; `uv run ruff check .` clean; `uv run pyright` or `mypy --strict`-ish on public API.
- Edge cases covered: empty DB, unicode, huge outputs, streaming replies, malformed JSON from LLMs, tool
  call argument strings that are not JSON, concurrent writers, WebSocket disconnect mid-turn, bad audio upload,
  provider timeout, missing keys, re-running mine/bench idempotently, Harbor export re-run overwriting cleanly.
- `touchstone demo && touchstone mine --provider scripted && touchstone bench run -m scripted` works with
  no keys, no network, no Docker.
