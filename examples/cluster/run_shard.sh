#!/bin/bash
# Generate one shard of a large data set with flashsim.
#
# Shard i gets its own seed, BASE_SEED + i, and its own event ids, i * EVENTS ... (i + 1) * EVENTS - 1, so that the
# shards can be combined with `flashsim merge`. The scheduler scripts in this directory call this script with the
# shard index; it also runs by hand, from the repository root:
#
#     bash examples/cluster/run_shard.sh 0                  # shard 0 with the settings below
#     EVENTS=1000 bash examples/cluster/run_shard.sh 3      # shard 3 with 1000 events
#     bash examples/cluster/run_shard.sh 0 --save-hits      # extra options go to flashsim generate
#
# ADAPT TO YOUR CLUSTER: the settings below (or set them in the environment of the job), in particular OUT_DIR
# (a work or scratch file system with enough space, not your home directory) and VENV.
set -euo pipefail

SHARD=${1:?usage: run_shard.sh SHARD_INDEX [extra flashsim generate options]}
shift

EVENTS=${EVENTS:-100000}        # events per shard
BASE_SEED=${BASE_SEED:-1000}    # shard i uses the seed BASE_SEED + i
PRESET=${PRESET:-full}          # physics preset
DEVICE=${DEVICE:-auto}          # auto, cpu or cuda
THREADS=${THREADS:-4}           # CPU threads; use the number of cores you requested
OUT_DIR=${OUT_DIR:-data/shards} # where the shards go
VENV=${VENV:-.venv}             # virtual environment with prob-reco-sbi installed

# Batch systems may start a job with an empty environment (HTCondor does, unless told otherwise).
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$OUT_DIR"
OUT=$(printf '%s/%s_shard%04d.h5' "$OUT_DIR" "$PRESET" "$SHARD")

# A shard that is complete already is skipped, so that a failed array can simply be submitted again.
# (flashsim writes to OUT.part first and renames it only when the file is complete.)
if [[ -f "$OUT" ]]; then
    echo "shard $SHARD: $OUT exists, nothing to do"
    exit 0
fi

echo "shard $SHARD on $(hostname): $EVENTS events, preset $PRESET, seed $((BASE_SEED + SHARD)), output $OUT"
flashsim generate --preset "$PRESET" -n "$EVENTS" \
    --seed $((BASE_SEED + SHARD)) --first-event-id $((SHARD * EVENTS)) \
    --device "$DEVICE" --threads "$THREADS" --no-progress -o "$OUT" "$@"
