#!/bin/bash
# Run a Touchstone dataset (or single task) through Harbor on the Mac mini's Docker, for humans.
#
# This is a thin wrapper over `touchstone bench` so there is ONE code path (run.py, which is tested):
# with no local Docker daemon and TOUCHSTONE_HARBOR_HOST set, run.py rsyncs the dataset (and, for a
# custom agent, the touchstone repo) to the host, runs Harbor there over SSH, and rsyncs jobs back.
#
# Usage: scripts/harbor-mini.sh -m <provider/model> [--agent packaged|replica] [--dataset <path>]
#   scripts/harbor-mini.sh -m openai/gpt-4o-mini --dataset touchstone
set -euo pipefail
export TOUCHSTONE_HARBOR_HOST="${TOUCHSTONE_HARBOR_HOST:-100.64.110.35}"
exec uv run touchstone bench "$@"
