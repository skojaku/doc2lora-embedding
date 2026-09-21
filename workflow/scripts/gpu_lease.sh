#!/usr/bin/env bash
# Lease one GPU with enough FREE memory, pin CUDA_VISIBLE_DEVICES, run the command, release.
#
# Unlike exps/2026-06-09-s2and/gpu_run.sh (which requires a GPU to be nearly idle, <6 GB *used*),
# this leaser looks at memory.FREE, so it still works when another user parks a resident service
# (e.g. an ollama server) on every card.  Locks live in a shared lock dir so parallel Snakemake
# jobs never land on the same GPU.
#
#   NEED_MB      minimum free VRAM required (default 20000)
#   GROUPC_GPUS  candidate GPU indices     (default "0 1 2 3")
#   LEASE_DIR    lock directory            (default /tmp/doc2lora_gpu_locks)
set -uo pipefail
LOCKDIR="${LEASE_DIR:-/tmp/doc2lora_gpu_locks}"; mkdir -p "$LOCKDIR"
CANDIDATES="${GROUPC_GPUS:-0 1 2 3}"
NEED="${NEED_MB:-20000}"
while true; do
  for g in $CANDIDATES; do
    free=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
    free=${free// /}
    # reclaim a stale lock whose holder died
    if [ -d "$LOCKDIR/gpu$g" ]; then
      p=$(cat "$LOCKDIR/gpu$g/pid" 2>/dev/null || echo "")
      if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then rm -rf "$LOCKDIR/gpu$g"; fi
    fi
    if [ "${free:-0}" -ge "$NEED" ] && mkdir "$LOCKDIR/gpu$g" 2>/dev/null; then
      echo $$ > "$LOCKDIR/gpu$g/pid"
      trap 'rm -rf "$LOCKDIR/gpu'"$g"'" 2>/dev/null' EXIT
      export CUDA_VISIBLE_DEVICES=$g
      echo "[gpu_lease] GPU $g (free ${free} MB) <- $*" >&2
      "$@"; rc=$?
      rm -rf "$LOCKDIR/gpu$g" 2>/dev/null; trap - EXIT
      exit $rc
    fi
  done
  sleep 10
done
