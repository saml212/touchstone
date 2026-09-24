# Touchstone

A touchstone is the black stone assayers rubbed gold against to prove its purity before anyone paid
for it. Touchstone does that for LLM agents: **one line of code captures what your agent actually
does**, the traces become a benchmark your stakeholders agree is fair, and the benchmark proves
whether a cheaper or open model is good enough **before** you switch or train.

Everything runs locally on a laptop — no cloud service, no paid service, no keys required. Real
providers, speech, and GPU training are all additive. By Pebble ML.

**Files are the source of truth for what you author; the database keeps what is captured.** Tasks,
checks and benchmarks live as diffable files beside `.touchstone/`; a task directory *is* a Harbor
task, so `harbor run -p tasks/<name>` runs it unchanged.

```
touchstone.toml            project config
checks.toml                policies: checks that apply to every task (mined + interview "always")
tasks/<name>/              one Harbor task per directory
  task.toml                Harbor 1.4 schema + [metadata.touchstone] + [[…check]] blocks (read aloud)
  instruction.md           the prompt; context.json + reference.json are the machine input + oracle
  environment/Dockerfile   solution/solve.sh (oracle)   tests/{test.sh,verify.py}
benchmarks/<name>.toml     a named set: tasks = […] or glob + tags
.touchstone/touchstone.db  captured traces (episodes, spans), the run record, interview rooms
.touchstone/runs/<id>/     run.json + results.jsonl — the portable copy of what the DB indexes
```

## Install

From GitHub (private, over SSH):

```bash
uv pip install git+ssh://git@github.com/saml212/touchstone       # into the current environment
uv tool install git+ssh://git@github.com/saml212/touchstone      # as a standalone `touchstone` CLI
```

Python 3.12+. The only hard dependencies are the web/CLI stack and two tiny check libraries; provider
SDKs, `faster-whisper`, ART, and TRL are optional extras you add when you need them.

## The one line

```python
import touchstone; touchstone.trace()   # traces land in ./.touchstone/touchstone.db
```

That patches whichever of `openai` (chat completions and the Responses API) / `anthropic` is
importable and records every model call and tool call — content parts, thinking/reasoning, refusals,
tool-call ids, stop reasons, cached tokens and per-call cost — without ever raising into your app.
Wrap a run in `with touchstone.episode("name"): ...` and mark the result with
`touchstone.outcome(score, label)`; `@touchstone.tool` captures a tool (and links it to the model
call that requested it), and `record_llm_call(...)` captures a turn you make yourself. Already
instrumented with OpenInference? `touchstone.trace(otel=True)` (extra `touchstone[otel]`) ingests
those OTel spans into the same store.

## The loop

```
import touchstone; touchstone.trace()   # 1. capture what your agent does
touchstone serve                        # 2. local UI: episodes, checks, tasks, rooms, scoreboard, train
touchstone mine                         # 3. agent reads traces + code, writes task dirs + proposals
touchstone interview <task>             # 4. voice/text room; stakeholders turn opinions into checks
touchstone bench run BENCH -m <spec>    # 5. prove a candidate against the incumbent
touchstone sample BENCH --student <spec> --teacher <spec>   # 6. find where the candidate fails
touchstone distill BENCH --student <spec>                   # 7. package those failures to train on
touchstone train prepare BENCH          # 8. SFT / preference / RL datasets, ready for GPU infra
harbor run -p tasks/                    # 9. the tasks are already Harbor tasks (on a Docker host)
```

Real output from the built-in zero-key demo (`touchstone demo` runs a scripted support agent):

```
$ touchstone mine --no-llm
KIND             SEV   SUP  RATIONALE
tool_not_called  hard    4  escalate used in 100% of bad vs 0% of good episodes
tool_called      hard    8  order_status used in 100% of good vs 0% of bad episodes
max_length       soft    8  p99 good output length 58 chars, +25% slack
no_pii           hard    1  email leaked in 1 output(s)
contains         soft    7  'all sorted is there anything else' appears in 88% of good o
contains         soft    7  'there anything else i can help' appears in 88% of good outp
proposed 6 check(s) (6 stats, 0 llm), cut 24 task(s)
```

`mine` reads tool calls and their results from the model spans' canonical messages (so it works for
the common app that only calls `touchstone.trace()` and never decorates its tools), proposes checks
**verifiable-first** — tool-call correctness, state assertions on tool arguments, schema validity,
safety — and stamps a `confidence` on each. Those programmatic ranks and the safety invariants are
written **enabled**; the statistical ranks (phrases, length) stay disabled pending review. `mine`
also writes a task directory for every recorded assistant turn. Each task is a **work queue** entry:
`active` (its reference is a valid oracle and an empty reply fails), `needs_checks` (an empty reply
already passes — add a check that measures the work), or `needs_solution` (the recorded reply fails
its own checks — a teacher or a human must supply one). Only `active` tasks enter a benchmark. Enable
the rest of the checks, `tasks sync` to re-materialise, freeze a benchmark, and run:

```
$ touchstone checks enable --all-mined
enabled 6 mined check(s)
$ touchstone tasks sync
synced 24 task(s)
$ touchstone bench create demo --all
wrote benchmarks/demo.toml (8 tasks)

$ touchstone bench run demo -m reference
MODELS
model      tasks  pass  rate%  errors  cost
---------  -----  ----  -----  ------  ----
reference  8      8     100.0  0       -
```

`reference` is the incumbent baseline — it replays each task's recorded reply, so it passes every
active task by construction (a check attaches only when the reference already passes it, and a task
counts only if its reference scores 1 and an empty reply scores 0). Prove a candidate against it
task-by-task with `touchstone bench proof <candidate-run> <incumbent-run>`.

## Sample and Distill

Two buttons close the loop between finding a gap and fixing it.

**Sample** runs a candidate (the *student*) on the benchmark, records its difficulty per task, and for
each active task it fails asks a *teacher* model for up to N perturbed variants — a paraphrased user
turn, a renamed tool, a tightened constraint. Each variant is a new task directory carrying
`parent_task` / `generated_by` provenance and a `generation.json` (the teacher, the method, the
SHA-256 of the parent context — hashes only, never contents — and the two validation rewards); a
variant is kept only if it passes the same oracle/nop gate. The student is re-run on the survivors,
and Sample reports the **frontier**: every task with `0 < pass_rate < 1` (the learnability band) plus
the ones only the incumbent passes.

```
touchstone sample demo --student openai:gpt-4o-mini --teacher claude-cli --variants 1
```

**Distill** packages exactly that frontier into training data: `train prepare` restricted to the
frontier, with teacher-verified demonstrations as the SFT targets and the *chosen* side of preference
pairs (over the student's failing replies), and the frontier tasks with their checks as the RL
verifier. For a `needs_solution` task the teacher's verified reply becomes the task's oracle. Sample →
Distill → Sample until the frontier empties or the pass rate stalls; the loop's stopping criteria are
an empty frontier, a below-threshold pass-rate delta, a teacher that fails the gate everywhere (which
opens interview rooms on the contested tasks), or a per-round cost cap.

```
touchstone distill demo --student openai:gpt-4o-mini
```

Difficulty is measured, not requested: the running pass rate per `(task, model)` lives in the SQLite
`difficulty` table and is cached into each `task.toml` for display. Judge checks are sampled N times
(default 3) and report an `agreement`; a judge criterion the model is unsure about (agreement below
`min_agreement`, default 0.67) never gates on its own.

The same tasks run under Harbor directly — `harbor run -p tasks -a oracle` replays each reference and
scores it against the checks in `task.toml`, with no export step.

## Checks

A check is `(kind, params)` plus routing metadata (`applies_to`, `severity`). A task passes when every
enabled **hard** check passes; **soft** checks are reported but never gate. Kinds, programmatic first:

`contains`, `not_contains`, `regex`, `not_regex`, `json_schema`, `tool_called`, `tool_not_called`,
`tool_order`, `max_length`, `min_length`, `no_pii`, `expr` (a sandboxed expression over
`output`/`tools`/`reference`), and `judge` (an LLM rubric; soft by default, skipped when no provider).
Prefer a programmatic kind that states the rule exactly; reach for `judge` only when none can.

Each check reads as a sentence — `name`, `rule`, `because`, `severity`, `source` — in a task's
`task.toml` or, when it applies to every task, in `checks.toml`. `checks.toml` holds **policies**
(with an `enabled` flag); `tasks sync` copies every enabled policy the reference gate passes into
each task with `source = "policy"`. `touchstone checks list|add|enable|disable|show|eval` manage
`checks.toml`; `tasks list|show|sync` inspect and rebuild task directories.

## Interview

`touchstone interview <task_id>` opens a room and prints (and opens) its URL; join `/rooms/<id>` from
any browser with a display name. The agent summarizes the task and the model's reply, asks one concrete
question at a time, and keeps a live draft-check list. Any participant can confirm — no host. A bare
"yes" commits the draft; a confirmation that carries an amendment ("yes, and one L is fine too") is
revised through the LLM first, then committed. A committed check is written as a
`[[metadata.touchstone.check]]` block in the task's `task.toml` — or, when the stakeholder says it
applies to every task, as an enabled policy in `checks.toml` — so `git diff` is the audit trail of
what everyone agreed to. Voice is push-to-talk (browser `MediaRecorder` → STT → message; agent reply → TTS to everyone);
text-only always works.

## Providers

Pass a `-m`/`--model` spec (repeatable) to `bench run`; the same specs drive mining and interviews.

| spec | what it is | key |
| --- | --- | --- |
| `scripted` | deterministic, for tests and the demo | none |
| `reference` | replays each task's recorded reply (the incumbent baseline) | none |
| `openai:<model>` | api.openai.com/v1 | `OPENAI_API_KEY` or Keychain |
| `openai-compatible:<base_url>:<model>` | any OpenAI-shaped `/v1` endpoint (vLLM, Ollama, …) | optional |
| `anthropic:<model>` | Anthropic Messages API | `ANTHROPIC_API_KEY` or Keychain |
| `claude-cli[:model]` | `claude -p` subprocess (Claude subscription) | none (uses your login) |
| `codex-cli[:model]` | `codex exec` subprocess (Codex subscription) | none (uses your login) |

Secrets resolve env-var first, then macOS Keychain
(`security find-generic-password -a sam -s <prefix><name> -w`). A secret is never printed and never
required. Run `touchstone doctor` for a one-table view of what's available (no installs, no network):

```
component             status   detail
--------------------  -------  --------------------------------------------
python                ok       3.12.x
database              ok       .touchstone/touchstone.db (exists)
sdk: openai           missing  not installed
provider: scripted    ok       always available
provider: reference   ok       always available (replays recorded references)
provider: claude-cli  ok       claude on PATH
speech: mode          ok       local — realtime needs OPENAI_API_KEY
speech: stt           ok       none — configured
tool: harbor          ok       /Users/you/.local/bin/harbor
tool: docker          missing  not on PATH
train: art            missing  not installed
train: trl            missing  not installed
```

## Config

`touchstone init` writes `touchstone.toml` and creates `.touchstone/`. Pass `--keychain-prefix` to pin
the Keychain service prefix; omit it to use the default `touchstone-`.

```toml
db_path = "./.touchstone/touchstone.db"
provider = "scripted"          # default judge/mining provider spec
agent_provider = "claude-cli"  # provider for mining LLM proposals + the interviewer
# keychain_prefix = "touchstone-"   # written only when you pass --keychain-prefix

[keychain]
openai = "openai-api-key"       # -> service "touchstone-openai-api-key"
anthropic = "anthropic-api-key"

[speech]
mode = "local"                 # local | realtime (OpenAI Realtime, needs OPENAI_API_KEY)
stt = "none"                    # none | faster-whisper | openai
tts = "browser"                 # browser | say | openai
# realtime_model = "gpt-realtime-2.1-mini"
# realtime_voice = "marin"
```

Every value also has an env override: `TOUCHSTONE_DB`, `TOUCHSTONE_PROVIDER`,
`TOUCHSTONE_AGENT_PROVIDER`, `TOUCHSTONE_KEYCHAIN_PREFIX`, `TOUCHSTONE_STT`, `TOUCHSTONE_TTS`,
`TOUCHSTONE_SPEECH_MODE` (env beats file beats default).

## Speech

Two modes, set by `[speech] mode`:

- **`local`** (default, zero-key, offline): push-to-talk → STT → text interviewer → TTS.
  STT: `faster-whisper` (local, `pip install 'touchstone[whisper]'`), `openai`
  (`gpt-4o-mini-transcribe`), or `none`. TTS: `browser` (`speechSynthesis`, zero-dep default),
  `say` (macOS), or `openai` (`gpt-4o-mini-tts`). Text-only always works.
- **`realtime`** (additive, needs `OPENAI_API_KEY`): one server-side OpenAI Realtime
  speech-to-speech session per room (`realtime_model`, default `gpt-realtime-2.1-mini`; `realtime_voice`,
  default `marin`). Browsers push-to-talk 24 kHz PCM over the room WebSocket and hear the agent's
  reply; several stakeholders share one session. The agent reads a check back in plain words before
  committing. On any Realtime error the room falls back to `local` for that session.

`touchstone doctor` reports the mode and whether realtime is reachable (key present), no network.

## Harbor

The tasks are already Harbor tasks — there is no export step. Each `tasks/<name>/` carries
`task.toml`, `instruction.md`, `context.json`, `reference.json`, `environment/Dockerfile`,
`solution/solve.sh`, and `tests/{test.sh,verify.py}`. The container verifier scores
`/app/output.json` against the `[[metadata.touchstone.check]]` blocks in `task.toml` with a
**vendored** copy of Touchstone's checks — no network, no `touchstone` install inside the container.
`judge` checks (which need an LLM) are excluded from the container copy.

```bash
touchstone export harbor            # prints the tasks path and the `harbor run` command
harbor run -p tasks -a oracle       # or a single task: harbor run -p tasks/<name>
touchstone bench harbor-run tasks/<name> -a <agent>   # shells out to harbor, or explains what's missing
```

If Docker or the `harbor` CLI is missing, `bench harbor-run` prints the exact command to run
elsewhere and exits non-zero — it never installs anything.

## Training hook

Touchstone never spends GPUs itself. It produces the datasets today and makes it obvious where infra
plugs in.

```bash
touchstone train prepare BENCH            # writes .touchstone/train/<bench>/
touchstone train submit  BENCH --backend null|art|trl
```

`prepare` writes (atomically; re-runs overwrite):

- `sft.jsonl` — canonical context + reference completion for tasks whose reference passes every
  attached hard check.
- `preference.jsonl` — `{prompt, chosen, rejected}` pairs: a passing candidate reply over a failing
  one (from run results), plus the reference over each failing candidate on tasks whose reference is
  good.
- `rl_tasks.jsonl` — task name, context, tools, and the serialized attached checks a reward
  verifier evaluates.
- `manifest.json` — counts + the target name.

`submit` picks a backend:

- `null` writes `train_plan.md` describing exactly what would run, and reports `planned`.
- `art` vendors the checks and writes a runnable `art_train.py` (an `art.TrainableModel` whose rollout
  replays a task and scores it with those checks as the reward).
- `trl` writes `trl_sft.yaml` + `run_trl.sh` for supervised fine-tuning on the passing references.

`art` and `trl` write their files and then stop with one sentence naming the file and command, e.g.:

```
$ touchstone train submit demo --backend trl
prepared datasets in .touchstone/train/demo
Wrote .touchstone/train/demo/trl_sft.yaml and .touchstone/train/demo/run_trl.sh; run
`pip install trl && bash run_trl.sh` in .touchstone/train/demo on a host with a GPU to train.
```

Copy the directory to a GPU box, `pip install` the backend, run the one command.

## Zero-key path

No keys, no network, no Docker — the whole loop works with the built-in `scripted` and `reference`
providers:

```bash
touchstone init
touchstone demo                        # 30 real episodes from a scripted support agent
touchstone mine --provider scripted    # or --no-llm for statistics only
touchstone checks enable --all-mined && touchstone tasks sync
touchstone bench create demo --all
touchstone bench run demo -m reference
touchstone train prepare demo
```

Add a real provider (`-m openai:gpt-4o-mini`, `-m openai-compatible:http://localhost:8000/v1:qwen`,
`-m claude-cli:sonnet`) and it slots into the same benchmark.

## Development

```bash
uv run pytest -q      # tests
uv run ruff check .   # lint
```

The design lives in `docs/DESIGN.md`.
