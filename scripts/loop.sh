#!/bin/zsh
# Karpathy-style experiment loop. Each iteration:
#   1. tests must pass and the tree must be clean;
#   2. a headless agent (no shell access) edits src/iaa/experiment.py and states its
#      hypothesis;
#   3. the driver reverts anything else the agent touched, runs the frozen harness
#      itself and records the result in results.tsv;
#   4. the driver commits a kept change or reverts a discarded one, and logs the
#      iteration in ai_log/.
# The agent never runs code, never writes the ledger and never submits to Kaggle.
#
# Usage: scripts/loop.sh [iterations]   (default 5)
set -u
ROOT="${0:A:h:h}"
cd "$ROOT" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
ITER="${1:-5}"
LOCK="$ROOT/.loop.lock"
LOG_DIR="$ROOT/ai_log"
HARNESS_TIMEOUT=2400
mkdir -p "$LOG_DIR"
DAYLOG="$LOG_DIR/loop_$(date +%Y-%m-%d).md"
EXP="src/iaa/experiment.py"

if ! mkdir "$LOCK" 2>/dev/null; then
  if [[ -f "$LOCK/pid" ]] && kill -0 "$(cat "$LOCK/pid")" 2>/dev/null; then
    echo "loop already running (pid $(cat "$LOCK/pid"))"; exit 0
  fi
  rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

# Read-only tools plus Edit. The driver reverts any edit outside experiment.py.
ALLOWED=("Read" "Grep" "Glob" "Edit")
PROFILE_ARGS=(${(z)$(llmp args coder)})

for i in $(seq 1 "$ITER"); do
  if ! .venv/bin/python -m pytest -q >/dev/null 2>&1; then
    echo "tests failing before iteration $i; stopping" | tee -a "$DAYLOG"; exit 1
  fi
  if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "working tree not clean before iteration $i; stopping" | tee -a "$DAYLOG"; exit 1
  fi
  ts=$(date +%Y-%m-%dT%H:%M:%S)
  out=$(claude -p "$(cat program.md)" "${PROFILE_ARGS[@]}" \
          --permission-mode acceptEdits --allowedTools "${ALLOWED[@]}" \
          --max-turns 30 --output-format text 2>&1)

  # Guardrail: revert every change except experiment.py (tracked and untracked).
  violations=$(git status --porcelain --untracked-files=all | awk '{print $NF}' | grep -vx "$EXP")
  if [[ -n "$violations" ]]; then
    git checkout -- . 2>/dev/null
    git restore --source=HEAD -- "$EXP" 2>/dev/null
    git clean -fdq
    verdict="violation"
    desc="agent edited files outside experiment.py: $(echo $violations | tr '\n' ' ')"
  elif echo "$out" | grep -q '^HYPOTHESIS: *NONE'; then
    git restore --source=HEAD -- "$EXP" 2>/dev/null
    echo "agent reports no open ideas left; stopping" | tee -a "$DAYLOG"
    git add "$DAYLOG"; git -c core.hooksPath=/dev/null commit -q -m "loop log: backlog exhausted" || true
    exit 3
  elif git diff --quiet -- "$EXP"; then
    verdict="no_change"
    desc="agent made no change"
  else
    verdict="pending"
    desc=$(echo "$out" | grep -m1 '^HYPOTHESIS:' | sed 's/^HYPOTHESIS: *//' | tr '\t' ' ' | cut -c1-240)
    [[ -z "$desc" ]] && desc="(no HYPOTHESIS line)"
  fi
  if [[ "$verdict" == "pending" ]] && ! perl -e 'alarm shift; exec @ARGV' 300 \
       .venv/bin/python -I -m iaa.harness --smoke x > "$LOG_DIR/last_run.txt" 2>&1; then
    # 20-second crash check on a subsample before spending a full run
    err=$(grep -E "Error|error" "$LOG_DIR/last_run.txt" | tail -1 | tr '\t' ' ' | cut -c1-160)
    verdict="crash"
    desc="SMOKE CRASH ($err): $desc"
    printf '%s\t%s\tcrash\t\t\t\t\t\t\t\t\t\t\t\t\t%s\t\n' "$ts" "$(git rev-parse --short HEAD)+dirty" "$desc" >> results.tsv
  elif [[ "$verdict" == "pending" ]]; then
    # The driver, not the agent, runs the harness and writes the ledger.
    if perl -e 'alarm shift; exec @ARGV' "$HARNESS_TIMEOUT" \
         .venv/bin/python -I -m iaa.harness "$desc" > "$LOG_DIR/last_run.txt" 2>&1; then
      verdict=$(tail -1 results.tsv | cut -f3)
    else
      verdict="crash"
      printf '%s\t%s\tcrash\t\t\t\t\t\t\t\t\t\t\t\t\t%s\t\n' "$ts" "$(git rev-parse --short HEAD)+dirty" "$desc" >> results.tsv
    fi
  fi

  row=$(tail -1 results.tsv)
  if [[ "$verdict" == "keep" ]]; then
    git add "$EXP" results.tsv
    git -c core.hooksPath=/dev/null commit -q -m "exp(keep): $desc" \
      -m "loop iteration $i, $ts; primary $(echo "$row" | cut -f4), min_fold $(echo "$row" | cut -f7)"
    action="committed $(git rev-parse --short HEAD)"
  else
    git restore --source=HEAD -- "$EXP"
    action="reverted experiment.py"
  fi
  {
    echo "## Iteration $i, $ts, verdict: $verdict"
    echo
    echo "Hypothesis: $desc"
    echo
    [[ "$verdict" == "keep" || "$verdict" == "discard" || "$verdict" == "over_budget" ]] && \
      echo "Ledger row: \`$row\`" && echo
    [[ "$verdict" == "crash" ]] && echo "Harness output:" && echo && tail -15 "$LOG_DIR/last_run.txt" | sed 's/^/    /' && echo
    echo "Agent report:"
    echo
    echo "$out" | tail -20 | sed 's/^/> /'
    echo
    echo "Driver action: $action"
    echo
  } >> "$DAYLOG"
  rm -f "$LOG_DIR/last_run.txt"
  if [[ "$verdict" != "keep" ]]; then
    git add results.tsv "$DAYLOG"
  else
    git add "$DAYLOG"
  fi
  git -c core.hooksPath=/dev/null commit -q -m "loop log: iteration $i $verdict" || true
  echo "iteration $i: $verdict  $desc"
done
