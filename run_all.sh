#!/usr/bin/env bash
# Full reproduction pipeline.  Steps that need SUMO are skipped when SUMO is not available,
# in which case the stored SUMO policies and results are used.
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${SUMO_HOME:-}" ] || python -c "import sumo" 2>/dev/null; then HAS_SUMO=1; else HAS_SUMO=0; fi

python tests.py

for s in 0 1 2; do
  for m in PPO PPO+CBF CertiFlow CertiFlow-noSTL CertiFlow-noCoop CertiFlow-noACI; do
    python train.py --method "$m" --seed "$s"
  done
  python train.py --method PPO --dr --seed "$s"
  python train.py --method CertiFlow --dr --seed "$s"
  if [ "$HAS_SUMO" = 1 ]; then python sumo_train.py --init CertiFlow-DR --seed "$s"; fi
done

python evaluate.py
python evaluate_dr.py
python evaluate_main.py
python exp_sweeps.py
python exp_extra.py phase

if [ "$HAS_SUMO" = 1 ]; then
  python sumo_eval.py --scenario ring --episodes 10
  python sumo_eval.py --scenario highway --episodes 3
  python exp_extra.py runtime residuals sumo_traces
fi

python make_tables.py
python make_figures.py
python make_thumbnails.py
