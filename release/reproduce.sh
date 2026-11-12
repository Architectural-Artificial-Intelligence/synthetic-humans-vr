#!/usr/bin/env bash
# Re-creates the analysis outputs from the shipped data. Run from the release root with the
# environment from requirements.txt active. Bootstrap B=2000 -> takes a while.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
B=reproduced/20261101_103808
mkdir -p "$B" reproduced
$PY analysis/robust_inference.py --out "$B/summary" --mixed
$PY analysis/stability.py --out "$B/stability"
$PY analysis/paired_baseline.py --out "$B/summary"
$PY analysis/demographics_table.py          # writes results/demographics_table.csv (coarsened occupation)
$PY analysis/make_paper_outputs.py --batch "$B"
$PY analysis/figure_stats.py --batch "$B"
$PY analysis/make_main_figures.py --batch "$B"
$PY analysis/gender_analysis.py --out reproduced/gender
$PY analysis/gender_supplement.py --gender reproduced/gender --batch "$B"
echo "done -> reproduced/"
