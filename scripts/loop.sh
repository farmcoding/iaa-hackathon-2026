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
#        LOOP_MARKET=emerging scripts/loop.sh [iterations]   (Part 2: experiment_emerging.py,
#        program_emerging.md, results_emerging.tsv)
set -u
ROOT="${0:A:h:h}"
cd "$ROOT" || exit 1
# /usr/sbin is needed: joblib (glum, scikit-learn) calls sysctl to count CPU cores
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"
export LOKY_MAX_CPU_COUNT=8
ITER="${1:-5}"
LOCK="$ROOT/.loop.lock"
LOG_DIR="$ROOT/ai_log"
HARNESS_TIMEOUT=4200  # two full runs when a keep is confirmed
# Stop with margin instead of at 0 bytes: on 25 Oct the disk filled mid-loop and
# took Docker's engine down with it. df's "available" excludes purgeable space.
MIN_FREE_GB=3
# coreutils timeout (brew coreutils): kills the harness's whole process group,
# loky workers included, and escalates to KILL. Exit 124 on timeout.
TIMEOUT=(/opt/homebrew/bin/gtimeout -k 30)
mkdir -p "$LOG_DIR"
DAYLOG="$LOG_DIR/loop_$(date +%Y-%m-%d).md"
MARKET="${LOOP_MARKET:-developed}"
if [[ "$MARKET" == "emerging" ]]; then
  EXP="src/iaa/experiment_emerging.py"; PROGRAM="program_emerging.md"
  RESULTS="results_emerging.tsv"; MODULE="iaa.experiment_emerging"
else
  EXP="src/iaa/experiment.py"; PROGRAM="program.md"; RESULTS="results.tsv"; MODULE="iaa.experiment"
fi
# harness commands only: the test suite runs on the developed default
HARNESS=(env IAA_MARKET="$MARKET" .venv/bin/python -I -m iaa.harness --module "$MODULE")

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

free_gb() { df -Pk "$ROOT" | awk 'NR == 2 { print int($4 / 1048576) }'; }
stop_if_low_disk() {
  local free; free=$(free_gb)
  if (( free < MIN_FREE_GB )); then
    echo "only ${free} GB free (margin ${MIN_FREE_GB} GB) $1; stopping" | tee -a "$DAYLOG"
    git restore --source=HEAD -- "$EXP" 2>/dev/null
    git add "$DAYLOG"; git -c core.hooksPath=/dev/null commit -q -m "loop log: stop on low disk (${free} GB free)" || true
    exit 4
  fi
}

for i in $(seq 1 "$ITER"); do
  stop_if_low_disk "before iteration $i"
  # queue behind any other full-dataset run (explain, analysis scripts) so the
  # smoke and harness timeouts below measure work, not waiting; see runguard.py
  .venv/bin/python -I -c 'from iaa import runguard; runguard.wait_free()'
  if ! .venv/bin/python -m pytest -q >/dev/null 2>&1; then
    echo "tests failing before iteration $i; stopping" | tee -a "$DAYLOG"; exit 1
  fi
  if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "working tree not clean before iteration $i; stopping" | tee -a "$DAYLOG"; exit 1
  fi
  ts=$(date +%Y-%m-%dT%H:%M:%S)
  # where the model to beat misses (data/cache/diagnostics.md, gitignored); step 1 of program.md
  IAA_MARKET="$MARKET" .venv/bin/python -I scripts/diagnostics.py >/dev/null 2>&1 || echo "diagnostics failed" > data/cache/diagnostics.md
  out=$(claude -p "$(cat "$PROGRAM")" "${PROFILE_ARGS[@]}" \
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
    desc=$(echo "$out" | grep -m1 '^HYPOTHESIS:' | sed 's/^HYPOTHESIS: *//' | tr '\t' ' ' | cut -c1-2000)
    [[ -z "$desc" ]] && desc="(no HYPOTHESIS line)"
  fi
  # Undefined names (a re-applied diff missing its import) cost eight iterations on
  # 19 Oct; catch them in a second and give the agent one fix attempt.
  if [[ "$verdict" == "pending" ]]; then
    lint=$(ruff check --select F821 --output-format concise "$EXP" 2>&1)
    if (( $? != 0 )); then
      claude -p "Your edit to $EXP fails a lint check for undefined names:
$lint
Fix only these (usually a missing import at the top of the file). Change nothing else." \
        "${PROFILE_ARGS[@]}" --permission-mode acceptEdits --allowedTools "${ALLOWED[@]}" \
        --max-turns 8 --output-format text >/dev/null 2>&1
      lint=$(ruff check --select F821 --output-format concise "$EXP" 2>&1)
      if (( $? != 0 )); then
        verdict="crash"
        desc="UNDEFINED NAME ($(echo "$lint" | grep -m1 F821 | sed 's/.*F821 //' | cut -c1-120)): $desc"
        printf '%s\t%s\tcrash\t\t\t\t\t\t\t\t\t\t\t\t\t%s\t\n' "$ts" "$(git rev-parse --short HEAD)+dirty" "$desc" >> "$RESULTS"
      fi
    fi
  fi
  smoke_rc=0
  if [[ "$verdict" == "pending" ]]; then
    # 20-second crash check on a subsample before spending a full run
    "${TIMEOUT[@]}" 300 \
      "${HARNESS[@]}" --smoke x > "$LOG_DIR/last_run.txt" 2>&1
    smoke_rc=$?
  fi
  if (( smoke_rc != 0 )); then
    if (( smoke_rc == 124 )); then
      err="timed out after 300 s on 50k rows per line"
    else
      err=$(grep -v -E "^\s*$|Warning|warn" "$LOG_DIR/last_run.txt" | tail -1 | tr '\t' ' ' | cut -c1-200)
      if (( smoke_rc == 139 )); then
        err="segmentation fault (exit 139), no Python traceback: often two OpenMP runtimes loaded together, see program.md"
      elif (( smoke_rc > 128 )); then
        err="killed by signal $(( smoke_rc - 128 )) (exit $smoke_rc)${err:+: $err}"
      fi
      [[ -z "$err" ]] && err="exit code $smoke_rc, no error text"
    fi
    verdict="crash"
    desc="SMOKE CRASH ($err): $desc"
    printf '%s\t%s\tcrash\t\t\t\t\t\t\t\t\t\t\t\t\t%s\t\n' "$ts" "$(git rev-parse --short HEAD)+dirty" "$desc" >> "$RESULTS"
  elif [[ "$verdict" == "pending" ]]; then
    stop_if_low_disk "before the full run of iteration $i"
    # The driver, not the agent, runs the harness and writes the ledger.
    if "${TIMEOUT[@]}" "$HARNESS_TIMEOUT" \
         "${HARNESS[@]}" "$desc" > "$LOG_DIR/last_run.txt" 2>&1; then
      verdict=$(tail -1 "$RESULTS" | cut -f3)
    else
      verdict="crash"
      printf '%s\t%s\tcrash\t\t\t\t\t\t\t\t\t\t\t\t\t%s\t\n' "$ts" "$(git rev-parse --short HEAD)+dirty" "$desc" >> "$RESULTS"
    fi
  fi

  row=$(tail -1 "$RESULTS")
  if [[ "$verdict" == "keep" ]]; then
    git add "$EXP" "$RESULTS"
    git -c core.hooksPath=/dev/null commit -q -m "exp(keep): $desc" \
      -m "loop iteration $i, $ts; primary $(echo "$row" | cut -f4), min_fold $(echo "$row" | cut -f7)"
    action="committed $(git rev-parse --short HEAD)"
    # a property experiment flips FROZEN_LINES; restore the motor-focus default after a keep
    # (19-20 Oct: left inverted, it turned 25 motor experiments into no-ops). The keep's saved
    # predictions cover every line, so the next run's frozen lines are this keep's.
    default='FROZEN_LINES: tuple[str, ...] = ("buildings", "contents")'
    if [[ "$MARKET" == "developed" ]] && ! grep -qxF "$default" "$EXP"; then
      sed -i '' -E "s/^FROZEN_LINES: tuple\[str, \.\.\.\] = .*/$default/" "$EXP"
      git add "$EXP"; git -c core.hooksPath=/dev/null commit -q -m "loop: restore FROZEN_LINES motor-focus default after keep"
      action="$action; FROZEN_LINES restored"
    fi
  else
    # keep the discarded change, so it can be reviewed or re-run on fresh seeds
    patch_file=""
    if ! git diff --quiet -- "$EXP"; then
      mkdir -p "$LOG_DIR/diffs"
      patch_file="$LOG_DIR/diffs/${ts//:/}_${verdict}.patch"
      git diff -- "$EXP" > "$patch_file"
    fi
    git restore --source=HEAD -- "$EXP"
    action="reverted experiment.py${patch_file:+ (diff saved to ${patch_file#$ROOT/})}"
  fi
  {
    echo "## Iteration $i, $ts, verdict: $verdict"
    echo
    echo "Hypothesis: $desc"
    echo
    [[ "$verdict" == "keep" || "$verdict" == "discard" || "$verdict" == "over_budget" ]] && \
      echo "Ledger row: \`$row\`" && echo
    [[ "$verdict" == "crash" ]] && echo "Harness output (warnings removed):" && echo && \
      grep -v -E "Warning|warnings.warn" "$LOG_DIR/last_run.txt" | tail -25 | sed 's/^/    /' && echo
    echo "Agent report:"
    echo
    echo "$out" | tail -20 | sed 's/^/> /'
    echo
    echo "Driver action: $action"
    echo
  } >> "$DAYLOG"
  rm -f "$LOG_DIR/last_run.txt"
  if [[ "$verdict" != "keep" ]]; then
    git add "$RESULTS" "$DAYLOG"
    [[ -d "$LOG_DIR/diffs" ]] && git add "$LOG_DIR/diffs"
  else
    git add "$DAYLOG"
  fi
  git -c core.hooksPath=/dev/null commit -q -m "loop log: iteration $i $verdict" || true
  echo "iteration $i: $verdict  $desc"
done
