#!/usr/bin/env bash
# Run every registry model as its own vlm_prompt.py call (one lane per provider,
# lanes in parallel, models within a lane sequential), verifying after each model.
# Resumable: finished models are skipped by the collector. Logs: data/runs/logs/.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
RUNS=${RUNS:-10}
mkdir -p data/runs/logs

lane() {
  provider=$1; workers=$2
  for m in $($PY - "$provider" <<'PYEOF'
import sys, vlm_prompt as v
print(" ".join(k for k, s in v.ALL_MODELS.items() if s.provider == sys.argv[1]))
PYEOF
  ); do
    for attempt in 1 2 3; do
      $PY vlm_prompt.py --models "$m" --runs "$RUNS" --workers "$workers" \
          >> "data/runs/logs/$m.log" 2>&1
      if $PY vlm_prompt.py --models "$m" --runs "$RUNS" --verify \
          >> "data/runs/logs/$m.log" 2>&1; then
        echo "$(date +%T) OK   $m (attempt $attempt)" >> data/runs/logs/_progress.log
        break
      fi
      echo "$(date +%T) RETRY $m (attempt $attempt)" >> data/runs/logs/_progress.log
      [ "$attempt" = 3 ] && echo "$(date +%T) FAIL $m" >> data/runs/logs/_progress.log
    done
  done
  echo "$(date +%T) LANE DONE $provider" >> data/runs/logs/_progress.log
}

lane openai 5 &
lane anthropic 5 &
lane google 5 &
lane deepinfra 10 &
wait
echo "$(date +%T) ALL DONE" >> data/runs/logs/_progress.log
