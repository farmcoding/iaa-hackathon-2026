#!/bin/zsh
# Run loop batches until the agent reports the backlog is exhausted (or MAX_BATCHES).
# After each batch: build the HTML dashboard (if the generator exists) and commit it.
# Pauses between 18:40 and 19:20 so the tree is clean when the Sunday 19:00 PR sweep
# runs, and never starts a batch in that window.
#
# Usage: scripts/run_batches.sh [max_batches] [iterations_per_batch]
#        (LOOP_MARKET=emerging passes through to loop.sh; the Part 1 dashboard is skipped)
set -u
ROOT="${0:A:h:h}"
cd "$ROOT" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"
export LOKY_MAX_CPU_COUNT=8
MAX="${1:-4}"
PER="${2:-5}"

in_sweep_window() { local hm=$(date +%H%M); (( 10#$hm >= 1840 && 10#$hm < 1920 )); }

# batch numbers continue from the last dashboard snapshot, so reruns don't overwrite them
START=$(ls reports/dashboard/after_batch_*.html 2>/dev/null | sed -E 's/.*_([0-9]+)\.html/\1/' | sort -n | tail -1)
START=$(( ${START:-0} + 1 ))
for b in $(seq "$START" $(( START + MAX - 1 ))); do
  while in_sweep_window; do sleep 120; done
  echo "== batch $b ($(date +%H:%M))"
  scripts/loop.sh "$PER"
  rc=$?
  if [[ "${LOOP_MARKET:-developed}" == "developed" && -f scripts/dashboard.py ]]; then
    .venv/bin/python -I scripts/dashboard.py --batch "$b" && \
      git add reports/dashboard && \
      git -c core.hooksPath=/dev/null commit -q -m "dashboard: after batch $b" || true
  fi
  (( rc == 3 )) && { echo "backlog exhausted"; break; }
  (( rc != 0 )) && { echo "loop stopped with code $rc"; break; }
done
