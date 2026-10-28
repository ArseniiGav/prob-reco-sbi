#!/usr/bin/env bash
# Run every reference solution in order, on the CPU, from the repository root:
#
#     bash solutions/stage1_flash/run_all.sh
#
# Each step writes its log to $LOG_DIR (default solutions/stage1_flash/logs, ignored by git; change it with
# LOG_DIR=... bash solutions/stage1_flash/run_all.sh), the run stops at the first error and prints the time per step and in total.
# Data sets are generated on first use and cached in solutions/stage1_flash/data (about 1.5 GB), so a second run is much faster.
# The optional GPU timing of T0.2 (python solutions/stage1_flash/task0_timing.py --gpu) is not part of this script: it needs a CUDA
# build of PyTorch and a visible GPU (see solutions/stage1_flash/README.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR=${LOG_DIR:-solutions/stage1_flash/logs}
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=""   # every solution runs on the CPU; the reference numbers are CPU numbers

steps=(
  "task0_setup.py"
  "task0_timing.py"
  "task0_timing.py --threads"
  "task0_inspect.py"
  "task1_prior.py"
  "task2_response.py"
  "task2_proposals.py"
  "task3_1_ladder.py"
  "task3_2_expected_charge.py"
  "task3_3_processes.py"
  "task3_4_naive_estimators.py"
  "task4_event_displays.py"
  "task4_animation.py"
  "task5_fluctuations.py"
  "task5_degeneracy.py"
  "task5_detector_design.py"
  "task5_detector_effects.py"
  # "make_readme_figures.py"  # NOT run: the public README figures (docs/img/detector.png, flash_photons.gif) stay as
  #                             # published in v0.1.0; run it by hand only to redraw them on purpose
  "make_answers.py"
  "make_notebooks.py"
)

start=$(date +%s)
for step in "${steps[@]}"; do
  tag=${step// /_}; tag=${tag//.py/}; tag=${tag//--/}
  log="$LOG_DIR/run_all_${tag}.log"
  script=${step%% *}
  args=${step#"$script"}
  t0=$(date +%s)
  printf '%s  %-30s ' "$(date +%H:%M:%S)" "$step"
  # shellcheck disable=SC2086  # $args holds the (unquoted on purpose) command line options of the step
  if python "solutions/stage1_flash/$script" ${args:-} > "$log" 2>&1; then
    echo "ok   $(( $(date +%s) - t0 )) s"
  else
    echo "FAILED after $(( $(date +%s) - t0 )) s (see $log)"
    exit 1
  fi
done
echo "total $(( $(date +%s) - start )) s"
