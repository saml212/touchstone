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
  checks/           dsl.py (Check dataclass + kinds + flat TOML surface), run.py (evaluate), judge.py
  tasks.py          read/write task directories — the authored source of truth
  policies.py       checks.toml (policies) + materialize() with the reference gate
  _verify.py        vendored into each task's tests/verify.py (reads checks from task.toml)
  mine/             miner.py (statistical + LLM proposals -> checks.toml; writes task dirs), cut.py
                    (episode -> Task builder), codebase.py (find system prompts / tool schemas in a repo)
  interview/        rooms.py (state), agent.py (question policy -> checks written to files), speech.py
  bench/            benchmark.py (resolve benchmarks/<name>.toml to task dirs), runner.py (replay tasks,
                    async, retries), report.py (scoreboard, proof table), harbor_run.py
  train/            trainer.py (Protocol + NullTrainer), datasets.py (sft/preference/rl exports),
                    art.py, trl.py (adapters that write configs + commands; raise clearly when no infra)
  server/           app.py (FastAPI), routes/*.py (files for tasks/checks/benchmarks), static/
  cli/              Typer: init, doctor, demo, serve, mine, checks, tasks, interview, bench, export, train
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
- `runs(id, target, model_spec, started_at, finished_at, meta JSON)` — `target` is a benchmark
  name, a `tasks/` path, or a glob.
- `results(run_id, task, passed INT, reward REAL, check_results JSON, output JSON, latency_ms,
  cost_usd, error)` — `task` is the task dir name; `check_results` is keyed by check name.
- `rooms(id, task_id, topic, created_at, closed_at)`, `room_messages(id, room_id, speaker, role, text, audio_path, ts)`
- Tasks, checks and benchmarks are **files**, not rows — see "v2 — contracts" below. `task_id` on a
  room is the task directory name.
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
3. Cutting: every recorded assistant turn in every episode becomes a replay task directory (not only
   the final turn). Context = messages before the cut + tools; reference = recorded assistant message.
   `policies.materialize` decides which checks land in each `task.toml` via the reference gate: a
   non-safety check attaches only when the reference passes it, safety checks
   (`no_pii`, `not_contains`, `not_regex`, `tool_not_called`) always attach, and failure tasks carry
   safety checks only. See "v2 — contracts" for the file layout.
Proposals are appended to `checks.toml` as `enabled = false` policies until a human (or an interview)
enables them; `tasks sync` re-materialises. Idempotent: re-mining does not duplicate policies, and
task directory names are stable (derived from the episode + span).

## Interview

A room is attached to a task (or a failure set). Participants join `/rooms/<id>` from any browser with a
display name; messages broadcast over WebSocket. The agent (LLM) opens with a summary of the task and the
model's output, asks one concrete question at a time ("What should it have done instead?", "Is that a hard
rule or a preference?", "Would this wording pass?"), maintains a draft check list visible to everyone, and
commits a check when a participant confirms. The agent prefers a programmatic check kind that states
the rule exactly and reaches for `judge` only when no programmatic kind can express it. A bare "yes"
commits the current draft as-is; a confirmation that carries an amendment ("yes, and one L is fine
too") is revised through the LLM first and only then committed, so the stored check reflects the
amendment. A committed check is written as a `[[metadata.touchstone.check]]` block with
`source = "interview"` in the task's `task.toml`, or, when the participant says it applies to every
task, as an enabled policy in `checks.toml`; the draft lives in room state. Multiplayer: no host; any
participant can confirm; the agent addresses people by name and reconciles disagreement by asking the group.
Voice: browser push-to-talk (MediaRecorder) → `POST /rooms/{id}/audio` → STT → message. Agent replies →
TTS → audio to all clients. Providers: STT `faster-whisper` (local, optional extra), `openai` (whisper-1),
`none`; TTS `openai`, `say` (macOS), `browser` (speechSynthesis, zero-dep default). Auto-detect in
`doctor`. Text-only always works.

## Bench

A benchmark is `benchmarks/<name>.toml` (`tasks = [...]` or `glob` + `tags`); a run `target` is a
benchmark name, a `tasks/` path, or a glob. Runner resolves the target to active task directories,
replays each task's context to each candidate `model_spec`, records the reply as output, evaluates the
task's checks, and stores a `Result` with the real `reward` plus a portable `.touchstone/runs/<id>/`
copy. Async with a semaphore, per-call timeout, 2 retries on transport errors, cost from token counts ×
a small price table (unknown model → null cost, never a guess).
Report: pass rate per model, per check, per tag; proof table "candidate vs incumbent"; JSON + terminal
table + UI. Determinism: `scripted` provider yields identical results on rerun. The `reference`
model spec is the honest incumbent: it replays each task's recorded reply, so `bench run -m reference`
passes every task whose attached checks the reference satisfies (all non-failure tasks by construction,
since a non-safety check attaches only when the reference already passes it) — the baseline a cheaper
candidate is proven against, and a way to confirm attached checks are satisfiable.

Harbor: the tasks are already Harbor tasks, so there is no export step. `touchstone export harbor`
prints the tasks path and the `harbor run -p tasks` command. Each task's `tests/verify.py` (with a
vendored `touchstone_checks`) scores `/app/output.json` against the checks in `task.toml`, writing
`/logs/verifier/reward.txt`. `touchstone bench harbor-run tasks/<name>` shells out to `harbor run`
when Docker is available; otherwise it prints the exact command and where Docker is missing.

## Train

`Trainer` Protocol: `prepare(conn, root, target, out_dir) -> DatasetBundle` and
`submit(bundle, config) -> JobHandle`. `datasets.py` resolves the target to task dirs and writes, all
atomically (re-runs overwrite), under `.touchstone/train/<target>/`: `sft.jsonl` (canonical context +
reference completion for tasks whose reference passes every attached hard check), `preference.jsonl`
(per task, `{prompt, chosen, rejected}` pairs — a passing candidate reply over a failing one from run
results, plus the reference over each failing candidate on tasks whose reference is good),
`rl_tasks.jsonl` (task name, context, tools, serialized attached checks for the reward verifier), and
`manifest.json` (counts + target name). `NullTrainer.submit` writes `train_plan.md` and returns a
handle with status "planned". `art.py` writes `art_train.py` (an `art.TrainableModel` + a rollout that
replays a task and scores it with the vendored `touchstone_checks`); `trl.py` writes
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
`mine [--code PATH] [--provider SPEC]`, `checks list|add|enable|disable|show|eval` (over `checks.toml`),
`tasks list|show|sync`, `interview <task>` (prints room URL, opens browser),
`bench create|run|report|proof|runs|harbor-run`, `export harbor|atif`, `train prepare|submit`. Exit
codes non-zero on failure; errors are one clear sentence.

## Quality bar

- `uv run pytest -q` green; `uv run ruff check .` clean; `uv run pyright` or `mypy --strict`-ish on public API.
- Edge cases covered: empty DB, unicode, huge outputs, streaming replies, malformed JSON from LLMs, tool
  call argument strings that are not JSON, concurrent writers, WebSocket disconnect mid-turn, bad audio upload,
  provider timeout, missing keys, re-running mine/bench idempotently, Harbor export re-run overwriting cleanly.
- `touchstone demo && touchstone mine --provider scripted && touchstone bench run -m scripted` works with
  no keys, no network, no Docker.

---

# v2 — contracts (decided 2026-09-24)

The four research tracks (capture, voice, task generation, contracts) converged on one change:
**authored artifacts are files, captured data is rows.** Everything below supersedes the v1 sections
above where they conflict.

## Files are the source of truth for tasks, checks and benchmarks

```
touchstone.toml            # project config
checks.toml                # policies: checks that apply to every task (interview "always" rules, mined safety rules)
tasks/<name>/              # one Harbor task per directory; `harbor run -p tasks/<name>` runs it unchanged
  task.toml                # Harbor 1.4 schema + [metadata.touchstone] + [[metadata.touchstone.check]] blocks
  instruction.md           # human/agent-facing render of the context and "write your reply to /app/output.json"
  context.json             # canonical messages[] + tools[] to replay
  reference.json           # the recorded incumbent reply (the oracle answer)
  environment/Dockerfile
  solution/solve.sh        # oracle: writes reference.json to /app/output.json
  tests/test.sh            # runs tests/verify.py -> /logs/verifier/reward.txt + rewards.json
  tests/verify.py          # vendored evaluator (dsl.py + run.py) reading the checks from task.toml
benchmarks/<name>.toml     # tasks = ["tasks/a", ...] or glob = "tasks/*" + tags = [...]
.touchstone/touchstone.db  # traces (episodes, spans), run record (runs, results), interview rooms
.touchstone/runs/<id>/     # run.json + results.jsonl, the portable copy of what the DB indexes
```

A check block reads as a sentence and is a template fill for an agent:

```toml
[[metadata.touchstone.check]]
name     = "escalates angry customers"
rule     = "must call the tool escalate_to_human"
kind     = "tool_called"
tool     = "escalate_to_human"
severity = "hard"
source   = "interview"            # mined | interview | policy | manual
because  = "Every resolved angry-customer episode escalated; none of the unresolved ones did."
```

Params are flat per-kind keys (`values`, `mode`, `pattern`, `schema`, `tool`, `arguments_match`,
`order`, `max`, `min`, `pii`, `expr`, `rubric`); `Check.from_toml`/`to_toml` map them onto the v1
`(kind, params)` core so the evaluator does not change. `name`, `rule`, `because`, `severity`,
`source`, `applies_to`, `confidence` (0–1, how sure the miner is) are prose/routing fields.

**Policies.** `checks.toml` holds `[[check]]` blocks with the same shape. When tasks are written
(mine, generate, interview) every policy whose reference gate passes is copied into the task's
`task.toml` with `source = "policy"`, so each task directory is self-contained for Harbor. Editing a
policy and re-running `touchstone tasks sync` re-materializes.

**Reference gate.** A non-safety check is written into a task only if the task's `reference.json`
passes it; safety kinds (`no_pii`, `not_contains`, `not_regex`, `tool_not_called`) always apply;
failure tasks (bad outcome) carry safety checks only. Unchanged from v1, now applied at write time.

## SQLite keeps rows for what is captured or measured

- `episodes`, `spans` — traces. `spans.kind` ∈ {`model`, `tool`}; `spans.tool_call_id` links a tool
  span to its call; a model span's output carries `stop_reason`, `reasoning` (thinking blocks, when
  the provider returns them), `refusal`; usage carries `cached_tokens`; canonical message `content` is
  a string or a list of parts (`text`, `image`, `file`, `audio`) — never flattened.
- `runs` (`target` = benchmark name or tasks path, `model_spec`, timings), `results` (`task` = task
  dir name, `reward REAL`, `passed`, `check_results` keyed by check name, `output`, `latency_ms`,
  `cost_usd`, `error`).
- `rooms`, `room_messages` — interview session state. Committed checks are written to the task's
  `task.toml` (or to `checks.toml` when the stakeholder says "always"); `git diff` is the audit trail.
- Deleted: `tasks`, `checks`, `benchmarks`, `room_checks` tables and their CRUD.

## Task generation: verifiable first, then teacher–student

Ranked sources of verifiable checks, mined in this order and marked with `confidence`:
1. tool-call correctness against recorded tool results; 2. state assertions from tool outputs;
3. schema validity / structured output; 4. recorded business outcome; 5. deterministic string rules;
6. judge rubrics (last, soft, sampled N times with agreement reported).

Every task must pass the two gates from the Harbor task-generation RFC before it counts:
**oracle scores 1** (`reference` replay passes) and **nop scores 0** (an empty reply fails). Tasks
that fail either gate are kept with `status = "rejected"` and the reason; they never enter a benchmark.

Two buttons:
- **Sample** — run the student (candidate model) on the benchmark and on perturbed variants the
  teacher generates from failing tasks; the frontier is every task with `0 < pass_rate < 1` plus the
  proof table's "only incumbent passes". Writes new task directories with `[metadata.touchstone]
  parent_task`, `difficulty` (empirical pass rate), and `generation.json` provenance (teacher model,
  source hashes).
- **Distill** — `train prepare` restricted to the frontier: teacher-verified demonstrations and
  preference pairs from exactly those failures. Sample → Distill → Sample until the frontier empties
  or pass rate stalls. Interview rooms open on contested frontier tasks so a human ruling enters as a
  check.

## Voice: realtime mode

`[speech] mode = "local" | "realtime"`. `realtime` opens one OpenAI Realtime session per room,
server-side, with tools `draft_check`, `commit_check`, `show_task`, `next_task` that call the same
functions as the text interviewer. Browsers stream push-to-talk audio over the room WebSocket and
receive the agent's audio and panel updates; several stakeholders share one session. `local` is the
zero-key fallback and never goes away. The agent always reads a check back in plain words before
committing.

## Distribution

PyPI name `touchstone-bench` (`touchstone` is taken); the CLI stays `touchstone`.
`uv tool install touchstone-bench` / `uvx touchstone-bench demo` / `pip install touchstone-bench`.
Releases: tag → GitHub Actions → PyPI trusted publishing. Publishing makes the source public.
