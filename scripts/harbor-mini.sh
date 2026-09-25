#!/bin/bash
# Run a Touchstone dataset (or single task) through Harbor on the Mac mini's Docker, for humans.
#
# Harbor bind-mounts local directories and resolves a custom agent with plain importlib, so both the
# dataset and the touchstone repo must live on the mini in the same Python env that runs harbor. This
# rsyncs both, runs harbor there via `uvx --from harbor --with <touchstone>` (so a laptop code change
# reaches the mini in one rsync), and rsyncs the job directory back. Idempotent; prints its commands.
#
# Usage: scripts/harbor-mini.sh <dataset-or-task-dir> <agent> [model] [extra harbor args...]
#   scripts/harbor-mini.sh touchstone oracle
#   scripts/harbor-mini.sh touchstone touchstone.harbor.agent:TouchstoneAgent openai/gpt-4o-mini
set -euo pipefail

HOST="${TOUCHSTONE_HARBOR_HOST:-100.64.110.35}"
REMOTE_ROOT="${TOUCHSTONE_HARBOR_REMOTE_ROOT:-/Volumes/1TB_SSD/pebble}"
REMOTE_PATH_PREFIX='export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"'

path="${1:?dataset or task directory}"
agent="${2:?agent name or module:Class import path}"
model="${3:-}"
shift $(( $# < 3 ? $# : 3 )) || true
extra=("$@")

path="${path%/}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"

# A single task dir runs as-is; a dataset root runs its implicit tasks/ dataset.
if [ -f "$path/task.toml" ]; then sync_root="$path"; rel_run="."; else sync_root="$path"; rel_run="tasks"; fi
name="$(basename "$sync_root")"
remote="$REMOTE_ROOT/$name"

run() { echo "\$ $*"; "$@"; }

run rsync -az --delete "$sync_root/" "$HOST:$remote/"
with=()
if [[ "$agent" == *:* ]]; then
  run rsync -az --delete "$repo_root/" "$HOST:$REMOTE_ROOT/touchstone/"
  with=(--with "$REMOTE_ROOT/touchstone")
fi

model_arg=""; [ -n "$model" ] && model_arg="-m $model"
harbor_cmd="uvx --from harbor ${with[*]} harbor run -p $rel_run -a $agent $model_arg -o jobs -n 4 -y ${extra[*]}"
echo "\$ ssh $HOST 'cd $remote && $harbor_cmd'"
ssh "$HOST" "$REMOTE_PATH_PREFIX; cd $remote && $harbor_cmd"

mkdir -p jobs
run rsync -az "$HOST:$remote/jobs/" jobs/
echo "Job directories synced to ./jobs"
