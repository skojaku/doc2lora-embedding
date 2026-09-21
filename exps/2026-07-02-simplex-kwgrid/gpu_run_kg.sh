#!/usr/bin/env bash
# Lease one free GPU (lock-dir), pin CUDA_VISIBLE_DEVICES, run the command, release.
# Own copy of exps/2026-06-09-s2and/gpu_run.sh with a higher idle threshold: this
# box's baseline nvidia-smi memory.used sits at ~6.6-7.8GB even with nothing
# running (driver overhead), so the original script's <6000MB free-check never
# passes and it hangs in its retry loop forever. Threshold here is <15000MB
# (actual decode load runs ~26-28GB), own lock dir so it doesn't collide with
# s2and's concurrent use of the same GPUs.
set -uo pipefail
LOCKDIR=/tmp/kg_gpu_locks; mkdir -p "$LOCKDIR"
CANDIDATES="${KG_GPUS:-0 1 2 3}"
THRESH="${KG_GPU_MEM_THRESH:-15000}"
while true; do
  for g in $CANDIDATES; do
    used=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    used=${used// /}
    if [ -d "$LOCKDIR/gpu$g" ]; then
      p=$(cat "$LOCKDIR/gpu$g/pid" 2>/dev/null || echo "")
      if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then rm -rf "$LOCKDIR/gpu$g"; fi
    fi
    if [ "${used:-99999}" -lt "$THRESH" ] && mkdir "$LOCKDIR/gpu$g" 2>/dev/null; then
      echo $$ > "$LOCKDIR/gpu$g/pid"
      trap 'rm -rf "$LOCKDIR/gpu'"$g"'" 2>/dev/null' EXIT
      export CUDA_VISIBLE_DEVICES=$g
      echo "[gpu_run_kg] GPU $g <- $*" >&2
      "$@"; rc=$?
      rm -rf "$LOCKDIR/gpu$g" 2>/dev/null; trap - EXIT
      exit $rc
    fi
  done
  sleep 8
done
