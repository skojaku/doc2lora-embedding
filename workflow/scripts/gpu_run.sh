#!/usr/bin/env bash
# Lease one free GPU (lock-dir), pin CUDA_VISIBLE_DEVICES, run the command, release.
# Skips GPUs busy with external processes (>6 GB used) so parallel Snakemake jobs don't collide.
# Override candidate GPUs with S2AND_GPUS (default "0 3"; GPU1=ollama, GPU2=other user are auto-skipped by the mem check).
set -uo pipefail
LOCKDIR=/tmp/s2and_gpu_locks; mkdir -p "$LOCKDIR"
CANDIDATES="${S2AND_GPUS:-0 1 2 3}"
while true; do
  for g in $CANDIDATES; do
    used=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 99999)
    used=${used// /}
    # reclaim stale lock if its holder died
    if [ -d "$LOCKDIR/gpu$g" ]; then
      p=$(cat "$LOCKDIR/gpu$g/pid" 2>/dev/null || echo "")
      if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then rm -rf "$LOCKDIR/gpu$g"; fi
    fi
    if [ "${used:-99999}" -lt 6000 ] && mkdir "$LOCKDIR/gpu$g" 2>/dev/null; then
      echo $$ > "$LOCKDIR/gpu$g/pid"
      trap 'rm -rf "$LOCKDIR/gpu'"$g"'" 2>/dev/null' EXIT
      export CUDA_VISIBLE_DEVICES=$g
      echo "[gpu_run] GPU $g <- $*" >&2
      "$@"; rc=$?
      rm -rf "$LOCKDIR/gpu$g" 2>/dev/null; trap - EXIT
      exit $rc
    fi
  done
  sleep 8
done
