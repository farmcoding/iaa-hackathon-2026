#!/bin/zsh
# TabM fits for the folds given as arguments, in order (e.g. A B final), on local Metal.
# Local GPU runs approved by the owner 2026-10-18; Kaggle GPUs need approval per run.
# Waits until no harness run or experiment loop is active (16 GB machine).
export PATH="/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
cd ~/.cache/iaa-gpu || exit 1  # GPU venv and outputs live here, outside the repo
while pgrep -f "iaa.harness|scripts/loop.sh|feature_study" >/dev/null; do sleep 30; done
for f in "$@"; do
  echo "start $f $(date +%H:%M)" >> out/tabm_folds.log
  .venv/bin/python -I ~/Projects/iaa-hackathon-2026/scripts/tabm_fit.py --fold "$f" --epochs 15 > "out/tabm_$f.log" 2>&1
  echo "end $f $(date +%H:%M) exit $?" >> out/tabm_folds.log
done
echo "all done $(date +%H:%M)" >> out/tabm_folds.log
