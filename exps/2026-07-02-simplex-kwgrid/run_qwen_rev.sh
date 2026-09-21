#!/bin/bash
export PYTHONPATH=$DOC_TO_LORA_SRC
export DOC2LORA_CKPT=data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin
export SRC_KW=1 KWTAG=_kwsrc ABS_MAXTOK=180 KW_MAXTOK=60 CELL_SAMPLE=18
for i in $(seq -w 99 -1 0); do
  [ "$i" = "00" ] && continue
  [ "$i" = "80" ] && continue
  s="strat${i}_mild"
  f="results/absfollow_${s}_kwsrc.json"
  # skip if already has >=18 cells (done by the forward worker)
  n=$(python -c "import json,os;p='$f';print(len(json.load(open(p))['cells']) if os.path.exists(p) else 0)" 2>/dev/null)
  [ "$n" -ge 18 ] 2>/dev/null && continue
  CUDA_VISIBLE_DEVICES=3 python decode_absfollow.py $s >> logs/allcurve_qwen_rev.log 2>&1
  echo "== done $s (rev) $(date +%H:%M) ==" >> logs/allcurve_qwen_rev.log
done
echo "QWEN_REV_COMPLETE" >> logs/allcurve_qwen_rev.log
