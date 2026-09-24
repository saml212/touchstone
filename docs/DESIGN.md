# Touchstone — design

A touchstone is the black stone assayers rubbed gold against to prove its purity before anyone paid
for it. Touchstone does that for LLM agents: one line of code captures what your agent actually does,
the traces become a benchmark your stakeholders agree is fair, and the benchmark proves whether a
cheaper or open model is good enough **before** you switch or train. Everything runs locally on a
laptop; real providers, speech and GPU training are additive. This is the engineer-facing doc;
`README.md` is the user-facing one. They must not contradict each other.

Brand: Touchstone (product) by Pebble ML. Aesthetic: cream `#FAF5E7`, brick `#8B2E1F`, hairlines,
Space Grotesk + JetBrains Mono, light mode.

## Non-negotiables

- Everything runs on a laptop. No cloud service, no paid service, no key is ever required. The
  zero-key demo (`touchstone demo`) produces real traces, mined checks, a benchmark and a scoreboard
  with the built-in `scripted`/`reference` providers. Real providers, speech and training are
  additive and slot into the same paths.
- Python 3.12+, `uv`-managed, one package, one SQLite file. No ORM, no Postgres. Docker is needed
  only to run the (already-Harbor) tasks in a container.
- Small and readable beats clever. Prefer stdlib; every dependency earns its place. Every module has
  pytest coverage of edge cases, not just happy paths.
- Secrets: env var first, then macOS Keychain (`security find-generic-password -a sam -s <name> -w`).
  Never print a secret; never require one.
- **Authored artifacts are files; captured data is rows.** Tasks, checks and benchmarks are diffable
  files; episodes, spans, runs, results, difficulty and rooms are SQLite rows.

## The loop

```
import touchstone; touchstone.trace()          # 1. capture what your agent does
touchstone serve                               # 2. local UI: episodes, checks, tasks, rooms, runs, train
touchstone mine                                # 3. agent reads traces + code, cuts tasks, proposes checks
touchstone interview <task>                    # 4. voice/text room; stakeholders turn opinions into checks
touchstone bench run <bench> -m <spec>         # 5. prove a candidate against the incumbent
touchstone sample <bench> --student <spec> --teacher <spec>   # 6. find where the candidate fails
touchstone distill <bench> --student <spec>                   # 7. package those failures to train on
touchstone train prepare <bench>               # 8. SFT / preference / RL datasets, ready for GPU infra
harbor run -p tasks/                            # 9. the tasks are already Harbor tasks (needs Docker)
```

## Package layout

```
touchstone/
  __init__.py       trace(), episode(), outcome(), tool(), record_llm_call()  — the public one-liners
  config.py         Settings from touchstone.toml + TOUCHSTONE_* env (env > file > default)
  store.py          SQLite (WAL) schema + typed dataclasses + every query. The only place SQL lives.
  messages.py       the ONE canonical message shape; canonical()/text_of()/to_openai()/to_anthropic()
  ids.py            sortable ULID-like ids (stdlib only)
  capture/          patch_openai.py (chat + Responses), patch_anthropic.py, litellm.py, spans.py,
                    context.py (episode ctx + @tool), openinference.py (OTel ingest), atif.py
  llm/              Provider protocol chat(messages, tools, json) -> Reply; scripted, reference, nop,
                    openai_compat, anthropic, claude_cli, codex_cli; registry.py resolves spec strings
  checks/           dsl.py (Check + kinds + the one param table), run.py (evaluate), judge.py
  tasks.py          read/write task directories — the authored source of truth; the work-queue gate
  policies.py       checks.toml policies + materialize() with the reference gate
  _verify.py        vendored into each task's tests/verify.py (reads checks from task.toml)
  mine/             stats.py (statistical proposals), llm.py (LLM proposals), miner.py (orchestrate),
                    cut.py (episode -> Task), codebase.py (find prompts / tool schemas in a repo)
  interview/        rooms.py (state + Hub), agent.py (question policy -> checks in files),
                    speech.py (STT/TTS), realtime.py (OpenAI Realtime bridge)
  bench/            benchmark.py (resolve benchmarks/<name>.toml), runner.py, report.py, harbor_run.py,
                    pricing.py (token -> cost)
  train/            trainer.py (Protocol + NullTrainer), datasets.py, art.py, trl.py
  loop/             frontier.py (difficulty + frontier + stopping), sample.py, teach.py, distill.py
  server/           app.py (FastAPI), routes/*.py, static/ (vanilla JS, no build step)
  cli/              Typer: init, doctor, demo, serve, mine, checks, tasks, interview, bench, sample,
                    distill, export, train
```

## Contracts: files are the source of truth

```
touchstone.toml            # project config
checks.toml                # policies: checks that apply to every task (interview "always", mined safety)
tasks/<name>/              # one Harbor task per dir; `harbor run -p tasks/<name>` runs it unchanged
  task.toml                # Harbor 1.4 schema + [metadata.touchstone] + [[…check]] blocks (read aloud)
  instruction.md           # the prompt; "write your reply to /app/output.json"
  context.json             # canonical messages[] + tools[] to replay
  reference.json           # the recorded incumbent reply (the oracle answer)
  environment/Dockerfile
  solution/solve.sh        # oracle: writes reference.json to /app/output.json
  tests/{test.sh,verify.py}  # vendored evaluator scoring /app/output.json against the checks
benchmarks/<name>.toml     # tasks = ["tasks/a", ...] or glob = "tasks/*" + tags = [...]
.touchstone/touchstone.db  # traces, run record, interview rooms
.touchstone/runs/<id>/     # run.json + results.jsonl — the portable copy of what the DB indexes
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
`order`, `max`, `min`, `pii`, `expr`, `rubric`); `Check.from_toml`/`to_toml` map them onto the
`(kind, params)` core the evaluator uses, so the evaluator never sees the prose. `name`, `rule`,
`because`, `severity`, `source`, `applies_to`, `confidence` (0–1, how sure the miner is) are
prose/routing fields. One table in `checks/dsl.py` is the single source for both the flat surface
and validation.

**Policies.** `checks.toml` holds `[[check]]` blocks with the same shape plus an `enabled` flag. When
tasks are written (mine, sample, interview) every enabled policy whose reference gate passes is
copied into the task's `task.toml` with `source = "policy"`, so each task directory is
self-contained for Harbor. Editing a policy and re-running `touchstone tasks sync` re-materialises;
interview/manual blocks already on a task are preserved.

**Reference gate.** A non-safety check is written into a task only if the task's `reference.json`
passes it; safety kinds (`no_pii`, `not_contains`, `not_regex`, `tool_not_called`) always apply;
failure tasks (bad outcome) carry safety checks only.

## The work queue and its gates

A task's `status` is a work queue, not a pass/fail flag; `tasks.validate` sets it at write time from
the two gates of the Harbor task-generation RFC — **oracle scores 1** (the `reference` replay passes)
and **nop scores 0** (an empty reply fails):

- `active` — both gates pass. Only these enter a benchmark.
- `needs_checks` — an empty reply already passes every hard check (the task measures nothing yet);
  invites an interview.
- `needs_solution` — the recorded reply fails its own hard checks (a failure with no oracle);
  invites a teacher demonstration.

`status_reason` records why. The empirical pass rate per `(task, model_spec)` lives in the SQLite
`difficulty` table and is cached into `[metadata.touchstone] difficulty` for display — difficulty is
measured, not requested.

## SQLite keeps rows for what is captured or measured

- `episodes(id, name, source, started_at, ended_at, outcome_score, outcome_label, meta)`.
- `spans(id, episode_id, parent_id, kind ['model'|'tool'], name, model, timings, input, output,
  tokens_in, tokens_out, cost_usd, error, tool_call_id)`. A model span's `input` is
  `{messages, tools, params}`; its `output` is `{message, stop_reason, usage}` with `stop_reason`
  normalized (`stop|tool_calls|length|content_filter|refusal|error`) and `usage` carrying
  `cached_tokens`/`cache_creation_tokens`/`reasoning_tokens` when reported. `cost_usd` is computed at
  capture from `bench/pricing.py` when the model is known. A tool span's `input` is `{name,
  arguments}`, `output` is `{result}`, and `tool_call_id` links it to the model call that requested it.
- `runs(id, target, model_spec, timings, meta)` — `target` is a benchmark name, a `tasks/` path, or a
  glob. `results(run_id, task, passed, reward, check_results, output, latency_ms, cost_usd, error)` —
  `check_results` is keyed by check name and carries a per-check `agreement` for sampled judge checks.
- `difficulty(task, model_spec, attempts, passes, pass_rate, updated_at)` — upserted by Sample.
- `rooms`, `room_messages` — interview session state. Committed checks are written to files;
  `git diff` is the audit trail.

Messages everywhere in the store are the ONE canonical shape (`messages.py`):
`{role, content: str | [{type: text|image|audio|file, …}], tool_calls?, reasoning?, refusal?,
tool_call_id?, name?}`. `content` is a string only when every part is text; multimodal parts are
never flattened; `arguments` is always a JSON string. `canonical()` accepts dicts and SDK objects
(OpenAI chat/Responses, Anthropic blocks) and is idempotent; `text_of()` flattens for checks/prompts;
the HTTP providers call `to_openai()`/`to_anthropic()` to replay. IDs are sortable ULID-like strings;
timestamps ISO-8601 UTC. SQLite is WAL with `busy_timeout`; each caller holds its own connection.

## Checks DSL

`Check(kind, params)` is evaluated against `Target(output_text, tool_calls, reference)`.
Kinds, programmatic first: `contains`, `not_contains`, `regex`, `not_regex`, `json_schema`,
`tool_called` (name + optional `arguments_match`), `tool_not_called`, `tool_order`, `max_length`,
`min_length`, `no_pii`, `expr` (a sandboxed expression over `output`/`tools`/`reference`/
`context_text` — the last being the flattened user/system text of the task, so a rule can refer to
what the user asked without pinning to a literal; it is written to each task's
`tests/context_text.txt` so the vendored container verifier reads it too), and
`judge` (an LLM rubric; soft by default, sampled `samples` times — default 3 — combined by majority
with a reported `agreement`, demoted to `passed=None` below `min_agreement`, default 0.67, and
skipped when no provider). Prefer a programmatic kind that states the rule exactly; reach for `judge`
only when none can. A task passes when every enabled **hard** check passes and none errored; **soft**
checks are reported, never gate. `regex`/`not_regex`/`arguments_match` patterns are rejected at
validation and refused at evaluation when a quantified group repeats unboundedly, so a pathological
pattern can never hang the engine.

## Mining

Input: episodes (optionally filtered/limited) + optional `--code <path>`. Ranked, verifiable-first,
each proposal stamped with `confidence`: (1) tool-call correctness against recorded tool results;
(2) state assertions from tool outputs; (3) schema validity; (4) recorded business outcome;
(5) deterministic string rules; (6) judge rubrics (last, soft, sampled). `mine/stats.py` does the
no-LLM ranks; `mine/llm.py` prompts the `agent_provider` (default `claude-cli`) for DSL proposals
with rationale and drops invalid ones with a logged reason; `mine/miner.py` orchestrates; `cut.py`
turns every recorded assistant turn into a replay task directory (`<episode-slug>-turn-<n>`).
Programmatic ranks and safety invariants are written **enabled**; statistical ranks stay disabled
pending review. Idempotent: re-mining does not duplicate policies and task names are stable.

## Bench

A benchmark is `benchmarks/<name>.toml` (`tasks = [...]` or `glob` + `tags`); a target is a benchmark
name, a `tasks/` path, or a glob — `benchmark.resolve` handles all three and yields **active** tasks
only. The runner replays each task's context to each candidate `model_spec`, records the reply,
evaluates the checks, and stores a `Result` plus a portable `.touchstone/runs/<id>/` copy. Async with
a semaphore, per-call timeout, 2 retries on transport errors; cost from token counts × the price
table (unknown model → null cost, never a guess). `reference` is the honest incumbent: it replays
each task's recorded reply, so `bench run -m reference` passes every active task by construction (a
non-safety check attaches only when the reference already passes it). `bench proof <candidate>
<incumbent>` diffs two runs task-by-task. The same tasks run under Harbor directly — there is no
export step; `touchstone export harbor` prints the tasks path and the `harbor run` command, and
`bench harbor-run <task_dir>` shells out to `harbor run` or prints the exact command when Docker is
missing.

## Sample and Distill

The flywheel between finding a gap and fixing it (`touchstone/loop/`, CLI `sample`/`distill`,
`POST /api/sample|distill`):

- **Sample** (`sample.py`) runs the student on the benchmark, records difficulty, and for each active
  task it fails asks the teacher for up to N perturbed variants (paraphrase, rename a tool, tighten a
  constraint). Each variant is a new task dir with `parent_task`/`generated_by` and a
  `generation.json` (teacher, method, the SHA-256 of the parent context — hashes only, never
  contents, and the oracle/nop validation); a variant is kept only if it passes the gate. The student
  is re-run on the survivors. The **frontier** is every active task with `0 < pass_rate < 1` plus the
  ones only the incumbent passes. Deterministic with `scripted` (a mechanical paraphrase when no
  teacher JSON parses).
- **Distill** (`distill.py`) is `train prepare` restricted to the frontier: teacher-verified
  demonstrations (`teach.py`, accepted only when they pass the task's hard checks, stored under
  `model_spec = "teacher:<spec>"`) as SFT targets and the chosen side of preference pairs, and the
  frontier tasks with their checks as the RL verifier. For a `needs_solution` task the verified demo
  becomes the task's `reference.json` (`reference_from`).

Sample → Distill → Sample until `frontier.check_stop` fires: an empty frontier, a below-threshold
pass-rate delta, a teacher failing the gate everywhere (which opens interview rooms on the contested
tasks), or a per-round cost cap.

## Interview

`touchstone interview <task>` opens a room and prints (and opens) its URL; anyone joins `/rooms/<id>`
with a display name. The agent (`interview/agent.py`) summarizes the task and the model's reply, asks
one concrete question at a time, and keeps a live draft-check list. No host: any participant can
confirm. A bare "yes" commits the draft; a confirmation carrying an amendment ("yes, and one L is
fine too") is revised through the LLM first, then committed. A committed check is written as a
`[[metadata.touchstone.check]]` block with `source = "interview"` in the task's `task.toml`, or as an
enabled policy in `checks.toml` when the stakeholder says it applies to every task — so `git diff` is
the audit trail. The agent prefers a programmatic kind and reaches for `judge` only when none fits.
A draft promoted to a policy is first generalised: any value copied verbatim from this task's
context (an order id, a name, an amount) is lifted by the LLM into a regex or an `expr` over
`context_text`, then validated to still pass this task's reference; a literal that cannot be
generalised is committed to this task only. After every commit the room's own task is
re-materialised through the reference gate, re-validated, and the room is told its new work queue in
one sentence; a policy commit also re-syncs every other task and reports how many it applied to.

## Speech

`[speech] mode = "local" | "realtime"`.

- **local** (default, zero-key, offline): browser push-to-talk → STT → text interviewer → TTS. STT:
  `faster-whisper` (extra `touchstone[whisper]`), `openai`, or `none`. TTS: `browser`
  (`speechSynthesis`, default), `say` (macOS), or `openai`. Text-only always works.
- **realtime** (additive, needs `OPENAI_API_KEY`): one server-side OpenAI Realtime session per room
  (`interview/realtime.py`) with tools `draft_check`/`commit_check`/`show_task`/`next_task` — the same
  `Interviewer` methods the text mode calls, one policy behind two front doors. Browsers stream
  push-to-talk 24 kHz PCM16 over the room WebSocket and hear the reply; several stakeholders share one
  session and the mic is a floor (first holder wins). `draft_check` returns a read-back the model must
  say before `commit_check`; committed checks and transcripts land in the store, so text stays the
  source of truth. On any OpenAI error the bridge posts one room message and falls back to local for
  that session (never a 500); a dropped socket reconnects once. GA wire shape: connect to
  `wss://api.openai.com/v1/realtime?model=…` with a bearer token, `session.update` carrying
  `type:"realtime"`, `output_modalities`, and a nested `audio.{input,output}` block with
  `format {type:"audio/pcm", rate:24000}` and `server_vad`.

`touchstone doctor` reports the mode and whether realtime is reachable — no installs, no network.

## Train

`Trainer` Protocol: `prepare(conn, root, target, out_dir) -> DatasetBundle` and
`submit(bundle, config) -> JobHandle`. `datasets.py` resolves the target to task dirs and writes,
atomically under `.touchstone/train/<target>/`: `sft.jsonl` (context + reference for tasks whose
reference passes every attached hard check), `preference.jsonl` (`{prompt, chosen, rejected}` pairs),
`rl_tasks.jsonl` (task, context, tools, serialized checks for the reward verifier), `manifest.json`.
Backends write files and stop with one sentence: `null` writes `train_plan.md` (status `planned`),
`art` writes `art_train.py` (an `art.TrainableModel` whose rollout replays a task and scores it with
the vendored checks), `trl` writes `trl_sft.yaml` + `run_trl.sh`. Touchstone never spends a GPU;
`distill` restricts `prepare` to the frontier. Copy the directory to a GPU box, `pip install` the
backend, run the one command.

## Server and CLI

FastAPI + uvicorn, single-page static UI (vanilla JS, no build step). Pages: Overview, Episodes,
Checks, Tasks, Rooms, Benchmarks, Train. JSON API under `/api/*`, WebSocket `/ws/rooms/{id}`;
everything the CLI does the API does, both calling the same functions. `touchstone` with no
arguments prints the loop in six lines and this project's next step — the same
`overview.next_step` hint the Overview page shows (and it never creates a database). CLI surface:
`init`, `doctor`, `demo`, `serve`, `mine`, `checks {list,add,enable,disable,show,eval}`,
`tasks {list,show,sync}`, `interview`, `bench {create,run,report,proof,runs,harbor-run}`, `sample`,
`distill`, `export {harbor,atif}`, `train {prepare,submit}`. Exit codes non-zero on failure; errors
are one clear sentence.

## Distribution

PyPI name `touchstone-bench` (`touchstone` is taken); the CLI stays `touchstone`.
`uv tool install touchstone-bench` / `uvx touchstone-bench demo` / `pip install touchstone-bench`.
Releases: tag → GitHub Actions → PyPI trusted publishing. Publishing makes the source public.

## Quality bar

`uv run pytest -q` green; `uv run ruff check .` clean; cognitive complexity ≤ 8 (`complexipy`). Edge
cases covered: empty DB, unicode, huge outputs, streaming replies, malformed LLM JSON, non-JSON tool
arguments, concurrent writers, WebSocket disconnect mid-turn, bad audio upload, provider timeout,
missing keys, re-running mine/bench/sync idempotently. The zero-key path —
`touchstone demo && mine --no-llm && bench create demo --all && bench run demo -m reference` — works
with no keys, no network, no Docker.
