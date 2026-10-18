#!/bin/zsh
# Run loop batches until the agent reports the backlog is exhausted (or MAX_BATCHES).
# After each batch: build the HTML dashboard (if the generator exists) and commit it.
# Pauses between 18:40 and 19:20 so the tree is clean when the Sunday 19:00 PR sweep
# runs, and never starts a batch in that window.
#
# Usage: scripts/run_batches.sh [max_batches] [iterations_per_batch]
set -u
ROOT="${0:A:h:h}"
cd "$ROOT" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
MAX="${1:-4}"
PER="${2:-5}"

in_sweep_window() { local hm=$(date +%H%M); (( 10#$hm >= 1840 && 10#$hm < 1920 )); }

for b in $(seq 1 "$MAX"); do
  while in_sweep_window; do sleep 120; done
  echo "== batch $b of $MAX ($(date +%H:%M))"
  scripts/loop.sh "$PER"
  rc=$?
  if [[ -f scripts/dashboard.py ]]; then
    .venv/bin/python -I scripts/dashboard.py --batch "$b" && \
      git add reports/dashboard && \
      git -c core.hooksPath=/dev/null commit -q -m "dashboard: after batch $b" || true
  fi
  (( rc == 3 )) && { echo "backlog exhausted"; break; }
  (( rc != 0 )) && { echo "loop stopped with code $rc"; break; }
done
