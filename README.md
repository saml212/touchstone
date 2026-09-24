# Touchstone

A touchstone is the black stone assayers rubbed gold against to prove its purity before anyone paid
for it. Touchstone does that for LLM agents: one line of code captures what your agent actually does,
the traces become a benchmark your stakeholders agree is fair, and the benchmark proves whether a
cheaper or open model is good enough **before** you switch or train. By Pebble ML.

Everything runs locally on a laptop — one SQLite file, no cloud, no keys required.

## Install & trace

```bash
pip install touchstone            # from GitHub for now
```

```python
import touchstone; touchstone.trace()   # traces land in ./.touchstone/touchstone.db
```

## The loop

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

## Zero-key demo

```bash
touchstone init
touchstone demo      # 30 real episodes from a built-in scripted support agent
touchstone doctor    # environment + available providers
```

## Status

**Stage 1**: capture core (episodes, spans, tools), SQLite store, scripted provider, ATIF export,
and the `init` / `doctor` / `demo` CLI. Mining, interview, bench, train, and the server land in
later stages.
