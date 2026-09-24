# Touchstone

A touchstone is the black stone assayers rubbed gold against to prove its purity before anyone paid
for it. Touchstone does that for LLM agents: **one line of code captures what your agent actually
does**, the traces become a benchmark your stakeholders agree is fair, and the benchmark proves
whether a cheaper or open model is good enough **before** you switch or train.

Everything runs locally on a laptop — one SQLite file, no cloud service, no paid service, no keys
required. Real providers, speech, Harbor export, and GPU training are all additive. By Pebble ML.

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

That patches whichever of `openai` / `anthropic` is importable and records every model call and tool
call. Wrap a run in `with touchstone.episode("name"): ...` and mark the result with
`touchstone.outcome(score, label)`; `@touchstone.tool` captures a tool, and `record_llm_call(...)`
captures a turn you make yourself.

## The loop

```
import touchstone; touchstone.trace()   # 1. capture what your agent does
touchstone serve                        # 2. local UI: episodes, checks, tasks, rooms, scoreboard, train
touchstone mine                         # 3. agent reads traces + code, proposes checks + cuts tasks
touchstone interview <task>             # 4. voice/text room; stakeholders turn opinions into checks
touchstone bench run BENCH -m <spec>    # 5. prove a candidate against the incumbent
touchstone train prepare BENCH          # 6. SFT / preference / RL datasets, ready for GPU infra
touchstone export harbor BENCH          # 7. tasks as a Harbor benchmark (`harbor run` on a Docker host)
```

Real output from the built-in zero-key demo (`touchstone demo` runs a scripted support agent):

```
$ touchstone demo
captured 30 episodes (18 resolved), 100 spans in .touchstone/touchstone.db

$ touchstone mine --no-llm
KIND             SEV   SUP  RATIONALE
tool_not_called  hard   12  escalate used in 100% of bad vs 0% of good episodes
tool_called      hard   18  order_status used in 100% of good vs 0% of bad episodes
max_length       soft   18  p99 good output length 59 chars, +25% slack
no_pii           hard    3  email leaked in 3 output(s)
contains         soft   15  'all sorted is there anything else' appears in 83% of good o
contains         soft   15  'there anything else i can help' appears in 83% of good outp
proposed 6 check(s) (6 stats, 0 llm), cut 60 task(s)
```

Everything mined is disabled until a human (or an interview) enables it. Enable, re-cut so the checks
attach to the tasks whose reference passes them, freeze a benchmark, and run:

```
$ touchstone checks enable --all-mined
enabled 6 mined check(s)
$ touchstone mine --no-llm            # re-cut: attach enabled checks to consistent tasks
$ touchstone bench create demo --all
created benchmark 01M38VG… (60 tasks)

$ touchstone bench run demo -m reference -m scripted
MODELS
model      tasks  pass  rate%  errors  cost
---------  -----  ----  -----  ------  ----
reference  60     45    75.0   0       -
scripted   60     42    70.0   0       -
```

`reference` is the incumbent baseline — it replays each task's recorded reply, so it passes every task
whose attached checks the recorded behaviour satisfied. Prove a candidate against it task-by-task:

```
$ touchstone bench proof <candidate-run> <incumbent-run>
both pass: 27   only incumbent: 18   only candidate: 15   both fail: 0
cost  incumbent: -   candidate: -
```

## Checks

A check is `(kind, params)` plus routing metadata (`applies_to`, `severity`). A task passes when every
enabled **hard** check passes; **soft** checks are reported but never gate. Kinds, programmatic first:

`contains`, `not_contains`, `regex`, `not_regex`, `json_schema`, `tool_called`, `tool_not_called`,
`tool_order`, `max_length`, `min_length`, `no_pii`, `expr` (a sandboxed expression over
`output`/`tools`/`reference`), and `judge` (an LLM rubric; soft by default, skipped when no provider).
Prefer a programmatic kind that states the rule exactly; reach for `judge` only when none can.

`touchstone checks list|add|enable|disable|show|eval`, and `tasks list|show|attach|detach`.

## Interview

`touchstone interview <task_id>` opens a room and prints (and opens) its URL; join `/rooms/<id>` from
any browser with a display name. The agent summarizes the task and the model's reply, asks one concrete
question at a time, and keeps a live draft-check list. Any participant can confirm — no host. A bare
"yes" commits the draft; a confirmation that carries an amendment ("yes, and one L is fine too") is
revised through the LLM first, then committed. Committed checks are attached to the task and enabled.
Voice is push-to-talk (browser `MediaRecorder` → STT → message; agent reply → TTS to everyone);
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
agent_provider = "codex-cli"   # provider for mining LLM proposals + the interviewer
# keychain_prefix = "touchstone-"   # written only when you pass --keychain-prefix

[keychain]
openai = "openai-api-key"       # -> service "touchstone-openai-api-key"
anthropic = "anthropic-api-key"

[speech]
stt = "none"                    # none | faster-whisper | openai
tts = "browser"                 # browser | say | openai
```

Every value also has an env override: `TOUCHSTONE_DB`, `TOUCHSTONE_PROVIDER`,
`TOUCHSTONE_AGENT_PROVIDER`, `TOUCHSTONE_KEYCHAIN_PREFIX`, `TOUCHSTONE_STT`, `TOUCHSTONE_TTS` (env
beats file beats default).

## Speech

STT: `faster-whisper` (local, `pip install 'touchstone[whisper]'`), `openai` (whisper-1), or `none`.
TTS: `browser` (`speechSynthesis`, zero-dep default), `say` (macOS), or `openai`. `touchstone doctor`
reports what's resolvable; text-only interviews always work.

## Harbor export

```bash
touchstone export harbor BENCH --out ./harbor-tasks
```

writes one task directory per task in Harbor's layout (`task.toml`, `instruction.md`,
`environment/Dockerfile`, `tests/test.sh`, `tests/test_outputs.py`, `solution/solve.sh`). The
container verifier scores `/app/output.json` with a **vendored** copy of Touchstone's checks — no
network, no `touchstone` install inside the container. `judge` checks (which need an LLM) are dropped.
Run them where Docker exists:

```bash
touchstone bench harbor-run ./harbor-tasks/<task-dir> -a <agent>
```

If Docker or the `harbor` CLI is missing, it prints the exact command to run elsewhere and exits
non-zero — it never installs anything.

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
- `rl_tasks.jsonl` — task id, context, tools, and the serialized attached checks a reward verifier
  evaluates.
- `manifest.json` — counts + the benchmark id.

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
touchstone checks enable --all-mined && touchstone mine --no-llm
touchstone bench create demo --all
touchstone bench run demo -m reference -m scripted
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
