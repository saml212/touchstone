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
touchstone bench -m <provider/model>     # run a model over the dataset; print pass rate per task
touchstone jobs                          # list Harbor job dirs with their pass rate per task
touchstone serve  /  touchstone interview  # open the review room UI
```

`touchstone bench` also takes `--agent packaged|replica`, `--dataset <dir>`, and
`--against <job_dir>` (a per-task comparison against a previous run). `touchstone doctor` reports the
environment. Capture is one line in your own app:

```python
import touchstone
touchstone.trace()   # records every model + tool call to .touchstone/touchstone.db
```

## The dataset

A survey produces a `touchstone/` directory in the customer's repo — a plain Harbor dataset they own
and version:

```
touchstone/
  dataset.toml      the Harbor dataset manifest (metadata; tasks run as the implicit tasks/ dataset)
  tasks/<name>/     one Harbor task per directory (instruction.md, task.toml, environment/,
                    solution/, tests/)
  environment/      the customer's system, copied to run in a sandbox
  agent/            the agent under test as a Harbor custom agent (agent.toml + tools.py)
  simulators/       a small local service per network boundary
  report.md         what was mapped, simulated, and left open
```

The agent under test is packaged as a Harbor custom agent
(`touchstone.harbor.agent:TouchstoneAgent`): an OpenAI-compatible tool-calling loop whose system
prompt comes from `agent/agent.toml` and whose tool schemas and dispatch come from `agent/tools.py`.
It records an ATIF `trajectory.json` for every run.

## Storage

Only what is captured lives in SQLite (`.touchstone/touchstone.db`): `episodes`, `spans`, `rooms`,
`room_messages`, and `reviews` (what a review room decided about a task's trial). Harbor's own
`jobs/` directories are the run record — Touchstone reads them, it has no runs table.

## Running Harbor (remote Docker note)

Harbor bind-mounts local directories, so it must run where a Docker daemon lives. A laptop without a
daemon can offload to a host: set

```toml
[harbor]
host = "100.64.110.35"           # an SSH host with Docker + harbor
remote_root = "/Volumes/1TB_SSD/pebble"
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
