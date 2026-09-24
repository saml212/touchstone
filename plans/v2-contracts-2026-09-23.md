# Touchstone v2 stage A — contracts inversion

Files are the source of truth for tasks, checks, benchmarks. SQLite keeps captured traces + run record.

Baseline: touchstone 8030 LOC, tests 4388, app.js 471 (total 12889). 352 tests green.

## Milestones (pytest -q + ruff green before every commit)

- [ ] M1 `checks/dsl.py`: Check gains `rule`, `because`, `source`, `confidence`; `from_toml`/`to_toml` flat per-kind surface. Evaluator untouched. + tests. (additive, green)
- [ ] M2 `touchstone/tasks.py`: Task dataclass; `write_task`/`read_task`/`list_tasks`/`task_name`; task dir layout (task.toml Harbor 1.4 superset, instruction.md, context.json, reference.json, environment/Dockerfile, solution/solve.sh, tests/{test.sh,verify.py,dsl.py,run.py,task.toml,reference.json}). sync preserve interview/manual. + tests.
- [ ] M3 `touchstone/policies.py`: read/write checks.toml; `materialize(task, policies)` + reference gate (moved from cut.py). + tests.
- [ ] M4 `bench/benchmark.py`: benchmarks/<name>.toml read/write + resolve (tasks list | glob+tags), active-only.
- [ ] M5 THE FLIP: store.py (drop tasks/checks/benchmarks/room_checks; Run.target; Result.task+reward; check_results keyed by name; SCHEMA_VERSION=2 migration); cut.py -> build_tasks; runner.py (target, reward, run files); report.py (key by dir+check name); harbor_export.py deleted -> verify vendoring + `export harbor` alias + `bench harbor-run`; mine (dirs + checks.toml disabled policies); interview (append to task.toml / checks.toml, drafts in room state); train/datasets.py (read dirs); server routes + _deps + app.js; cli. + all tests.
- [ ] M6 docs: README + delete wrong v1 DESIGN sections.
- [ ] M7 demo dogfood (mine --no-llm, checks enable --all-mined, tasks sync, bench create demo --all, bench run demo -m reference) + Mac mini harbor run -a oracle -n 4. Restart demo server.

## Key decisions
- `task_name(episode, span)` = `slug(episode.name)-<span_id.lower()>` (stable, sortable, filesystem-safe, deterministic).
- Container verifier reads only files under /tests. write_task copies task.toml (non-judge checks) + reference.json into tests/ (generated artifacts; root task.toml is authored SoT). verify.py reads tests/task.toml + tests/reference.json, grades /app/output.json -> /logs/verifier/reward.txt + rewards.json.
- reward.txt = 1.0/0.0 (proven Harbor format). judge checks excluded from tests/ copy (no LLM in container), kept in root task.toml.
- Gates: status active|rejected. oracle (reference passes its checks) must be 1, nop (empty reply) must be 0, else rejected + reason; benchmarks resolve active only.
- All mined proposals -> checks.toml disabled policies. Interview commits -> task.toml block (source=interview) or checks.toml (policy, when "always").
- spans.kind stays 'llm' (rename to 'model' is stage B).
