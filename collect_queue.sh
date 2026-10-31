#!/usr/bin/env bash
# Parallel model queue: each model = its own vlm_prompt.py call (+ verify, up to 3
# attempts). Skips models that are already complete or currently running.
# Usage: PAR=8 WORKERS=4 ./collect_queue.sh
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
export RUNS=${RUNS:-10} WORKERS=${WORKERS:-4}
mkdir -p data/runs/logs

one() {
  m=$1
  if pgrep -f -- "vlm_prompt.py --models $m " >/dev/null; then
    echo "$(date +%T) SKIP-RUNNING $m" >> data/runs/logs/_progress.log; return
  fi
  for attempt in 1 2 3; do
    $PY vlm_prompt.py --models "$m" --runs "$RUNS" --workers "$WORKERS" >> "data/runs/logs/$m.log" 2>&1
    if $PY vlm_prompt.py --models "$m" --runs "$RUNS" --verify >> "data/runs/logs/$m.log" 2>&1; then
      echo "$(date +%T) OK   $m (attempt $attempt)" >> data/runs/logs/_progress.log; return
    fi
    echo "$(date +%T) RETRY $m (attempt $attempt)" >> data/runs/logs/_progress.log
  done
  echo "$(date +%T) FAIL $m" >> data/runs/logs/_progress.log
}
export -f one; export PY

$PY - <<'PYEOF' | xargs -P "${PAR:-8}" -I{} bash -c 'one {}'
import vlm_prompt as v
done = {m for m in v.ALL_MODELS if v.verify_model(m, "main", 10)["complete"]}
# slow/expensive reasoning models first so they overlap with the quick ones
order = sorted((m for m in v.ALL_MODELS if m not in done),
               key=lambda m: -v.ALL_MODELS[m].max_tokens)
print("\n".join(order))
PYEOF
echo "$(date +%T) QUEUE DONE" >> data/runs/logs/_progress.log
