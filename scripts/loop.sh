#!/bin/zsh
# Karpathy-style experiment loop: one headless agent run per iteration, each editing
# only src/iaa/experiment.py, scored by the frozen harness. The driver (this script)
# owns git: it commits kept runs, reverts discarded ones and logs every iteration.
#
# Usage: scripts/loop.sh [iterations]   (default 5)
# Never submits to Kaggle. Submissions go through the human gate (docs/SIGNOFF.md).
set -u
ROOT="${0:A:h:h}"
cd "$ROOT" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
ITER="${1:-5}"
LOCK="$ROOT/.loop.lock"
LOG_DIR="$ROOT/ai_log"
mkdir -p "$LOG_DIR"
DAYLOG="$LOG_DIR/loop_$(date +%Y-%m-%d).md"

# single-instance lock (macOS has no flock): atomic mkdir + stale-PID takeover
if ! mkdir "$LOCK" 2>/dev/null; then
  if [[ -f "$LOCK/pid" ]] && kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
    echo "loop already running (pid $(cat "$LOCK/pid"))"; exit 0
  fi
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

ALLOWED=(
  "Read" "Grep" "Glob"
  "Edit(src/iaa/experiment.py)"
  "Bash(.venv/bin/python -I -m iaa.harness *)"
)
PROFILE_ARGS=(${(z)$(llmp args coder)})

for i in $(seq 1 "$ITER"); do
  # guardrail 1: frozen core and tests must pass before the agent runs
  if ! .venv/bin/python -m pytest -q >/dev/null 2>&1; then
    echo "tests failing before iteration $i; stopping" | tee -a "$DAYLOG"; exit 1
  fi
  if [[ -n "$(git status --porcelain -- src configs tests program.md)" ]]; then
    echo "working tree not clean before iteration $i; stopping" | tee -a "$DAYLOG"; exit 1
  fi
  before=$(wc -l < results.tsv)
  ts=$(date +%Y-%m-%dT%H:%M:%S)
  out=$(claude -p "$(cat program.md)" "${PROFILE_ARGS[@]}" \
          --permission-mode acceptEdits --allowedTools "${ALLOWED[@]}" \
          --max-turns 40 --output-format text 2>&1)
  # guardrail 2: nothing but experiment.py may change
  changed=$(git status --porcelain -- src configs tests program.md scripts | grep -v "src/iaa/experiment.py")
  if [[ -n "$changed" ]]; then
    git checkout -- src configs tests program.md scripts
    verdict="violation"
  elif (( $(wc -l < results.tsv) == before )); then
    verdict="no_run"
  else
    verdict=$(tail -1 results.tsv | cut -f3)
  fi
  row=$(tail -1 results.tsv)
  if [[ "$verdict" == "keep" ]]; then
    git add src/iaa/experiment.py results.tsv
    git -c core.hooksPath=/dev/null commit -q -m "exp(keep): $(tail -1 results.tsv | cut -f16)" \
      -m "loop iteration $i, $ts; primary $(echo "$row" | cut -f4), min_fold $(echo "$row" | cut -f7)"
  else
    git checkout -- src/iaa/experiment.py
    git add results.tsv
    git -c core.hooksPath=/dev/null commit -q -m "exp($verdict): $(echo "$row" | cut -f16)" || true
  fi
  {
    echo "## Iteration $i, $ts, verdict: $verdict"
    echo
    echo "Ledger row: \`$row\`"
    echo
    echo "Agent report:"
    echo
    echo "$out" | tail -20 | sed 's/^/> /'
    echo
    echo "Driver action: $( [[ $verdict == keep ]] && echo "committed $(git rev-parse --short HEAD)" || echo "reverted experiment.py, ledger row committed")"
    echo
  } >> "$DAYLOG"
  echo "iteration $i: $verdict"
done
