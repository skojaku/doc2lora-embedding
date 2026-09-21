#!/usr/bin/env bash
# Unattended ICAE pipeline: wait for pool-slot embedding -> train adapter -> embed every benchmark
# subset (self-scheduling across free GPUs) -> run all evals -> summarize. Safe to launch anytime;
# it blocks until the pool shards finish.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."   # repository root
set -a; source .env 2>/dev/null; set +a
export HF_HOME="$(pwd)/data/agent_assets/hf_cache" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export S2AND_GPUS="0 1 2 3"   # gpu_run auto-skips GPUs busy with other users (>6GB)
IC=exps/2026-06-21-icae-benchmark
KR=exps/2026-06-09-kron-adapter
S2=exps/2026-06-09-s2and
GPU="bash $S2/gpu_run.sh"
log(){ echo "[pipe $(date +%H:%M:%S)] $*"; }

# 1) wait for both pool shards
while pgrep -f "python $IC/embed_pool_slots.py" >/dev/null; do sleep 60; done
[ -f "$IC/pool_slots.npy" ] || { log "pool_slots.npy missing -> abort"; exit 1; }
log "pool done -> train adapter"

# 2) train invertible per-token adapter (1 GPU). Big batch amortizes the per-step matrix_exp(4096^2)
#    cost, so far fewer steps cover the same data. Skip if already trained.
if [ ! -f "$IC/adapter_icae.pt" ]; then
  BS=128 STEPS=3000 $GPU python $IC/train_icae_adapter.py > $IC/log_train.txt 2>&1
fi
[ -f "$IC/adapter_icae.pt" ] || { log "TRAIN FAILED (see log_train.txt)"; exit 1; }
log "adapter trained -> embed benchmark subsets"

# 3) embed all subsets; gpu_run leases a free GPU per job (others queue), so only ~2 load at once
SPECS=("field economics" "field psychology" "aps" "s2and zbmath" "s2and qian" \
       "s2and arnetminer" "s2and pubmed" "s2and kisti")
pids=()
for spec in "${SPECS[@]}"; do
  tag="${spec// /_}"
  ( BATCH=48 $GPU python $IC/embed_apply.py $spec ) > "$IC/log_embed_${tag}.txt" 2>&1 &
  pids+=($!); sleep 8
done
wait "${pids[@]}" || true
log "embedding done -> evals"

# 4a) field/APS evals (INCLUDE_ICAE adds icae + icae_genkron; fixed topic + negative pool)
pids=()
for f in economics psychology aps; do
  INCLUDE_ICAE=1 OUT_SUFFIX=_icae TOPIC_IDS_FILE=$IC/${f}_topic_ids.parquet \
    NP_FUT_POOL_FILE=$IC/${f}_fut_pool.parquet NP_COHORT_FILE=$IC/${f}_cohorts.parquet \
    $GPU python $KR/eval_all.py --field $f --enc qwen > $IC/log_eval_${f}.txt 2>&1 &
  pids+=($!); sleep 8
done
wait "${pids[@]}" || true

# 4b) S2AND evals
pids=()
for ds in zbmath qian arnetminer pubmed kisti; do
  OUT_SUFFIX=_icae KRON_STEPS=2000 KRON_REG=10 \
    $GPU python $S2/and_eval.py $ds qwen > $IC/log_eval_s2and_${ds}.txt 2>&1 &
  pids+=($!); sleep 8
done
wait "${pids[@]}" || true
log "evals done -> summarize"

# 5) summarize
python $IC/summarize.py > $IC/RESULTS.md 2>&1 || log "summarize failed"
log "ALL DONE"
