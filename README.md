# Touchstone

A touchstone is the black stone assayers rubbed gold against to prove its purity before anyone paid
for it. Touchstone does that for AI agents: it turns a company's running agent into a
[Harbor](https://github.com/harbor-framework/harbor) benchmark of its own work, lets the people who
know the product review that benchmark by voice, and turns the results into training data.

Harbor is the middle of the product, unchanged. Touchstone is four extensions around it:

1. **Capture** — one line of code records what the production agent does.
2. **Survey** — a read-only agent reads the recordings and the code, copies the system into a Harbor
   environment, and writes Harbor tasks against it.
3. **Review** — a voice/text room where product people look at finished trials and correct the
   verifiers; Harbor regrades after every correction.
4. **Train** — turns finished jobs into distillation and RL datasets.

Everything Touchstone produces is a plain Harbor dataset the customer owns and can run without us.
Local-first: it runs on the customer's machine with their own Claude or Codex login — no Touchstone
account, no Touchstone servers. Open source. By Pebble ML.

Horizontal by design: nothing in `touchstone/harbor/` knows about any one customer, product, or
domain.

## The five commands

```
touchstone init                          # write touchstone.toml + the .touchstone/ db
touchstone demo                          # run the built-in agent and capture episodes (zero-key)
touchstone survey <repo>                 # read-only: map the code, simulate its services, score fidelity
touchstone bench -m <provider/model>     # run a model over the dataset; print pass rate per task
touchstone jobs                          # list Harbor job dirs with their pass rate per task
touchstone serve  /  touchstone review     # open the review room (voice or text)
touchstone train                         # turn finished jobs into distill + RL datasets
```

`touchstone survey` reads the recordings and the code with a read-only coding agent
(`[survey] provider`, default `claude-cli`; the customer's login pays), maps the tools and their
network boundaries, generates a SQLite-backed simulator per service, and replays every recorded call
through the real tools to score each simulator's fidelity. It is idempotent — outputs are reused
unless `--force` — and touches nothing outside `touchstone/` in the target repo.

`touchstone bench` also takes `--agent packaged|replica`, `--dataset <dir>`, and
`--against <job_dir>` (a per-task comparison against a previous run). `touchstone doctor` reports the
environment. Capture is one line in your own app:

```python
import touchstone
touchstone.trace()   # records every model + tool call to .touchstone/touchstone.db
```

Run your real agent on a different model with one env var: set `TOUCHSTONE_MODEL` and capture rewrites
the `model=` keyword on every openai/anthropic/litellm call before it goes through — same SDK, same
code path. (A model passed positionally is left untouched.) This is how the packaged agent under test
is benched against a candidate model without editing the customer's code.

## The dataset

A survey produces a `touchstone/` directory in the customer's repo — a plain Harbor dataset they own
and version:

```
touchstone/
  dataset.toml      the Harbor dataset manifest (metadata; tasks run as the implicit tasks/ dataset)
  tasks/<name>/     one Harbor task per directory (instruction.md, task.toml, environment/,
                    solution/, tests/)
  environment/      the customer's system, copied to run in a sandbox
  agent/            the agent under test as a Harbor custom agent (agent.toml, tools.py, entry.py)
  simulators/       a small local service per network boundary
  baseline.json     what the current setup passes today (the first-five-minutes sentence)
  report.md         what was mapped, simulated, and left open
```

The agent under test runs in one of two modes, chosen automatically and recorded in `agent/agent.toml`:

- **packaged** (highest fidelity): the survey generates `agent/entry.py` — a `run(user_message)`
  that drives the customer's *real* agent loop for one message, importing their own modules. The
  Harbor agent runs `bash /app/agent/run.sh` in the sandbox with the model as a setting
  (`TOUCHSTONE_MODEL`) and imports the trajectory the customer's own capture wrote. An adapter check
  proves entry.py runs and calls a tool before this mode is chosen.
- **replica** (fallback when the code will not run): `touchstone.harbor.agent:TouchstoneAgent` runs
  an OpenAI-compatible tool-calling loop whose system prompt comes from `agent/agent.toml` and whose
  tool schemas + dispatch come from `agent/tools.py`.

Either way it records an ATIF `trajectory.json`, and `touchstone bench -m <candidate>` re-runs it on
any model without touching the customer's code.

## The review room

`touchstone serve` then `touchstone review` opens a room (voice or text) where a product person walks
the finished trials with an AI. It opens from the product's goal (the job labels and the baseline
pass count), then picks trials in order — verifier unsure, models disagree, never reviewed, then
gate failures — and for each reads the instruction, the trajectory in plain words, and every
criterion's score, and asks "do you agree it passed?". Agreement records trust (the share of
reviewed trials where the human agreed with the verifier, shown live). On a disagreement the agent
drafts a criterion change — edit/add/remove a rewardkit check, a dimension weight, a judge line, or
the instruction wording — reads it back, and on "yes" writes the `tests/` file and runs
`harbor job regrade` (the tasks are authored with a separate verifier so grading reruns from the
recorded artifacts, no agent), then reads out the new reward and any other trials that moved.
"Always" applies the same criterion to every task with the same job. Everything the room decides is
a row in `reviews` and a file change under `touchstone/` — nothing else.

## Training data

`touchstone train` reads the Harbor job directories (`trajectory.json` + `reward.json` per trial)
and writes, under `touchstone/train/`, `distill.jsonl` (full trajectories from trials that scored at
or above the threshold, for a student to copy), `rl_tasks.toml` (tasks in the learnability band —
pass rate strictly between 0 and 1 — with the verifier as the reward), and `manifest.json` (per-task
pass rate per model and where each task went). `touchstone train --teacher <spec>` runs the teacher
job first. Training itself stops at the exact GPU command; Touchstone produces the data, not the run.

## Storage

Only what is captured lives in SQLite (`.touchstone/touchstone.db`): `episodes`, `spans`, `rooms`,
`room_messages`, and `reviews` (what a review room decided about a task's trial). Harbor's own
`jobs/` directories are the run record — Touchstone reads them, it has no runs table.

## Running Harbor (remote Docker note)

Harbor bind-mounts local directories, so it must run where a Docker daemon lives. A laptop without a
daemon can offload to a host: set

```toml
[harbor]
host = "harbor-host.example"           # an SSH host with Docker + harbor
remote_root = "/srv/touchstone"
```

and `touchstone bench` (via `touchstone/harbor/run.py`) rsyncs the dataset to that host, runs Harbor
there over SSH, and rsyncs the job directory back. `scripts/harbor-mini.sh` is a thin wrapper for the
same path. With a local Docker daemon, everything runs locally and these settings are ignored.

## Install

```
uv tool install --from git+ssh://git@github.com/saml212/touchstone touchstone-bench
```

or, in a checkout, `uv run touchstone …`. Requires [Harbor](https://docs.harborframework.com)
(`uv tool install harbor`) and a Docker daemon to run benchmarks.

## Develop

```
uv run pytest -q
uv run ruff check .
```

Commits are small and sequential; see `docs/DESIGN.md` for the full v3 design.
