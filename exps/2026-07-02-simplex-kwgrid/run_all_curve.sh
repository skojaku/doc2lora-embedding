#!/bin/bash
# $1 = gpu, $2 = script, $3 = tag  ; stats mode, skips 00 & 80 (kept full for figures)
export PYTHONPATH=$DOC_TO_LORA_SRC
export DOC2LORA_CKPT=data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin
export SRC_KW=1 KWTAG=_kwsrc ABS_MAXTOK=180 KW_MAXTOK=60 CELL_SAMPLE=18
for i in $(seq -w 0 99); do
  [ "$i" = "00" ] && continue
  [ "$i" = "80" ] && continue
  s="strat${i}_mild"
  CUDA_VISIBLE_DEVICES=$1 python $2 $s >> logs/allcurve_$3.log 2>&1
  echo "== done $s ($3) $(date +%H:%M) ==" >> logs/allcurve_$3.log
done
echo "ALLCURVE_$3_COMPLETE" >> logs/allcurve_$3.log
