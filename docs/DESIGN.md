# Touchstone — design (v3, 2026-09-24)

Touchstone turns a company's running AI agent into a Harbor benchmark of its own work, lets the
people who know the product review that benchmark by voice, and turns the results into training
data. Harbor (harbor-framework/harbor) is the middle of the product, unchanged. Touchstone is four
extensions around it:

1. **Capture** — one line of code records what the production agent does.
2. **Survey** — a read-only agent reads the recordings and the code, copies the system into a
   Harbor environment, and writes Harbor tasks against it.
3. **Review** — a voice/text room where product people look at finished tasks and trials with an
   AI and correct the verifiers. Harbor regrades after every correction.
4. **Train** — a Harbor plugin that turns finished jobs into distillation and RL datasets.

Everything Touchstone produces is a plain Harbor dataset the customer owns and can run without us,
publish, or sell. Local-first: it runs on the customer's machine with their own Claude or Codex
login; no Touchstone account, no Touchstone servers. Open source. Paid later: training, hosting,
brokering environments to labs. Horizontal: nothing in the design names a customer.

## Vocabulary (Harbor's words where Harbor has one)

- **Episode** — one recorded production conversation, with an outcome (score 0–1, label).
- **Span** — one model call or tool call inside an episode. Messages are stored in one canonical
  shape (`touchstone/messages.py`). Episodes export to and import from ATIF (`trajectory.json`).
- **Task** — a Harbor task: `instruction.md`, `task.toml`, `environment/`, `solution/`, `tests/`.
  One task per job-to-be-done (e.g. "refund a double charge"), not per model reply. Each task's
  `environment/` carries a `Dockerfile` (`FROM` the one shared image) — Harbor discovers a task only
  when it has an `environment/` dir, and the build is a cache hit on the base. `task.toml`'s optional
  `[task]` registry ref is omitted (its name must be exactly `org/name`, which a per-task slug is
  not); provenance (episode ids, tools seen — never episode contents) lives under
  `[metadata.touchstone]`.
- **Environment** — the customer's system copied into a sandbox: their real code where it runs on
  its own, and a **simulator** for every service that crosses the network (orders DB, Stripe, CRM,
  email, Slack, a human). A simulator is a small service with the same interface backed by a local
  SQLite file, generated from the calling code, the service's public API docs and every recorded
  call/response. Every simulator carries a **fidelity score**: the share of recorded calls it
  reproduces when replayed.
- **Agent under test** — the customer's own agent packaged as a Harbor custom agent with the model
  as a setting (highest fidelity). Fallback: a **replica** agent rebuilt from the system prompt and
  tools seen in the recordings, used when there is no code. The model is swapped with one env var:
  `TOUCHSTONE_MODEL` — capture rewrites the `model=` keyword on every openai/anthropic/litellm call
  before it goes through (same SDK, same code path; a model passed positionally is left untouched),
  so `touchstone bench -m <candidate>` reruns the agent on any model without editing the code.
- **Simulated user** — Harbor's user agent plays the customer from a persona and goal drawn from
  real episodes; the agent under test never sees the goal.
- **Verifier** — rewardkit criteria in `tests/`: end-state checks against simulator databases
  first ("exactly one refund on A1094 for $106.37"), then tool-use checks, then judges. Dimensions:
  `correctness`, `safety` (a no-PII check on every task, so weights are uniform), `autonomy` (no
  human tool called), `quality` (judge, soft). Grading is a **separate verifier**
  (`[verifier] environment_mode = "separate"`) in a fresh env built from `tests/Dockerfile` (the
  tests baked in), with `[environment] network_mode = "public"` so it can `uvx` harbor-rewardkit.
  `task.toml` declares the **artifacts** the verifier reads (the trajectory, `output.json`, and each
  simulator's `state.db`), so `touchstone review` can `harbor job regrade` a corrected criterion
  from the recorded artifacts without rerunning any agent. A sidecar `tests/descriptions.toml`
  (keyed `"<file>:<index>"`) carries one plain-English sentence per criterion for the review room;
  the verifier never reads it.
- **Gates** — a task counts only when Harbor's oracle agent scores 1 and the nop agent scores 0.
- **Dataset** — `touchstone/` in the customer's repo: `dataset.toml`, `tasks/`, `environment/`,
  `agent/`, `simulators/`, `report.md`. Delivered as a pull request; the customer merges it.
  `dataset.toml` is metadata only — its `tasks` list stays empty, because Harbor's manifest pins
  tasks by published digest (published refs), so local runs use the implicit dataset
  (`harbor run -p <root>/tasks`).
- **Trial / Job** — Harbor's. Touchstone reads job directories; it has no runs table.
- **Review** — a room over one task and its trials. Verifier disagreement → criterion change →
  `harbor job regrade` → the room reads out what moved. A benchmark's **trust** = share of
  reviewed trials where the human agreed with the verifier.

## The customer's first five minutes

```
pip install touchstone-bench          # or uvx
import touchstone; touchstone.trace() # one line, run production for a while
touchstone survey .                   # read-only; builds touchstone/ ; opens a PR if in git
```
`survey` ends with one sentence about their product, never a folder listing:
"Built 40 tasks from 312 conversations. Your current setup passes 29. 11 failures — walk through
them? (touchstone review)". Then `touchstone bench -m <candidate>`, `touchstone review`,
`touchstone train`.

## Survey (the new part; read-only by construction)

Inputs: the trace DB, the repo (read-only git token or local checkout), optionally DB schemas and
API docs. Never production credentials, never production data beyond the recordings.
Runs with `Settings.agent_provider` (claude-cli/codex-cli; the customer's login pays).

Steps, each a function with a report line:
1. **Map** the code: entrypoints, system prompts, tool functions, the model call site, every
   network boundary (HTTP clients, DB drivers, SDKs), and the tools' schemas.
2. **Sort** each tool into *runs-on-its-own* (copied verbatim into the environment image) or
   *crosses-the-network* (needs a simulator).
3. **Simulate**: for each boundary, generate `simulators/<name>/` (FastAPI or the driver's wire
   protocol, SQLite-backed, seeded from scrubbed recordings) plus a **fidelity test**: replay every
   recorded call, compare responses, write `fidelity.json`. Below a threshold the simulator is
   flagged, never silently used. Generated simulators bind `127.0.0.1` only (never `0.0.0.0`), and a
   service whose base URL is a compile-time constant — one the replay cannot repoint at the local
   simulator — is flagged rather than risk a call to the real (production) service.
4. **Package the agent**: `agent/` = the customer's agent as a Harbor custom agent
   (`harbor.agents.BaseAgent`), the model call site rewired to a setting (`--model`). Fallback
   replica when there is no code.
5. **Group** episodes by job-to-be-done (LLM clustering over the first user turn + tools used),
   name each group in plain words, pick real cases as variants.
6. **Write tasks**: per group, `instruction.md` (the simulated user's goal, with a canary comment),
   `persona.md`, seed state for the simulators (PII scrubbed), `tests/` rewardkit criteria plus a
   `tests/descriptions.toml` sidecar of plain-English sentences (no file paths), and a
   `solution/solve.sh` replaying a recorded success. Instruction/persona/criteria are written by the
   survey provider; the criterion sentences fall back to the mechanical description on a bad answer.
7. **Gate** every task with oracle and nop via `harbor run`; tasks that fail a gate go to
   `report.md` under "needs review" with the reason, not into `dataset.toml`.
8. **Baseline**: run the packaged agent with its current model; write the first-five-minutes
   sentence and `report.md` (what was mapped, simulated, fidelity per simulator, open questions
   for the review room).
Re-running survey is idempotent and diff-friendly (stable names, sorted output); a GitHub Action
re-runs it on main and opens a PR.

Tiers by access: code + traces (highest fidelity) → traces only (simulators inferred from recorded
calls; replica agent) → docs/interview only (lowest, labelled). The fidelity score says which.

## Review

Starts from the product's main goal ("this agent exists to resolve support tickets without a
human"), then works task by task: the AI reads the task and the trial it chose (verifier unsure,
models disagree, or a human hasn't seen this task) and asks "do you agree it passed?". Agreement
records trust. Disagreement → the AI proposes a criterion change in `tests/`, Harbor regrades,
the AI reads out the new pass counts. Rules stated as "always" become shared criteria. Human
interventions, autonomy, and product judgment enter here, not in capture code. Speech: local
(push-to-talk → transcribe → text agent → TTS) or realtime (OpenAI Realtime, one session per
room). Multiplayer as before.

## Train

`touchstone train` reads Harbor job directories (`trajectory.json` + `reward.json` per trial):
- **distill.jsonl** — full multi-turn trajectories (tool calls included) from trials of a teacher
  model (or the customer's incumbent) that scored ≥ threshold; the student copies them.
- **rl_tasks.toml** — tasks in the learnability band for the student (0 < pass rate < 1 across
  its trials), with paths; the verifier is the reward.
- **manifest.json** — per-task pass rates per model and which file each task went to.
Per task: pass rate 0 → distill; 0–1 → RL; 1 → hold out. `touchstone train --teacher <spec>`
runs the teacher job first. Training itself stops at the GPU line with the exact command.

## Storage

- `.touchstone/touchstone.db` (SQLite): `episodes`, `spans`, `rooms`, `room_messages`, `reviews`
  (task, trial, verdict, speaker, ts). Nothing else. No tasks, checks, runs, results tables.
- `touchstone/` (files): the Harbor dataset, owned by the customer, versioned in their git.
- Harbor's own `jobs/` directories are the run record.

## Repository layout (target)

```
touchstone/
  __init__.py  config.py                               # capture entry + settings
  messages.py + messages_wire.py                        # one canonical shape; render back to wire
  store.py + store_models.py                            # the only SQL; the typed rows (no SQL)
  capture/     llm/     interview/ (speech, rooms, realtime)
  harbor/      dataset.py (layout, dataset.toml), run.py (harbor run wrapper), jobs.py (read
               trials/rewards/trajectories), atif.py (episode -> ATIF) + atif_import.py (ATIF ->
               episode), agent.py (packaged agent base + replica), rewardkit.py (criteria helpers)
  survey/      map.py sort.py simulate.py fidelity.py package.py group.py gate.py baseline.py
               report.py  survey.py (orchestrator)  tasks.py (task orchestrator) +
               task_text.py (provider-authored instruction/persona/criteria) + task_files.py
               (write the task dir)
  review/      agent.py (goal → trial → verdict → criterion change → regrade) + prompt.py (tools +
               system) + facts.py (dataset readers, shared with server/pages)
  train/       datasets.py plugin.py write.py
  server/      pages.py (+ pagedata.py) + routes + static (Overview, Tasks, Trials, Review, Train)
  cli/
```
Deleted from v2: policies.py, checks/, mine/, bench/, loop/, `runs/results/difficulty`; and in v3
the placeholder `interview/agent.py` + `touchstone interview`, superseded by the review agent.

## Quality bar

`uv run pytest -q`, `ruff`, complexity ≤ 8 on every function (CI runs `complexipy`). Every stage
live-verified with `harbor run` on the Mac mini's Docker (`DOCKER_HOST=ssh://100.64.110.35` from the
laptop); `touchstone/harbor/run.py` rsyncs the dataset to the host, runs Harbor over SSH, and rsyncs
the job directory back. A provider API key needed there is forwarded on **stdin** (`read -r` into an
env var), never on argv or in the printed command, so it never lands in a process list. The
survey agent has its own benchmark: the example repo, then two open-source agent repos; measured by
simulator fidelity, oracle/nop pass rates, and baseline agreement with recorded outcomes.
Zero-key path: `touchstone demo` records the built-in agent; survey on it with `scripted`
produces a gate-passing dataset with no network.
