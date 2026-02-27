#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
CKPT="checkpoints/openvla-7b-oft-finetuned-libero-object"
BASE=experiments/robot/libero/configs/base_prune.json
TMP=tmp_r3_th.json

OUT=logs/r3_threshold
mkdir -p $OUT

DELTA_METHODS=("net" "arc_sum" "hypot")
LEFF=(0.10 0.15 0.20)
POS_TH=(0.0 0.01)
ROT_TH=(0.0 0.02)

for DM in "${DELTA_METHODS[@]}"; do
  for LE in "${LEFF[@]}"; do
    for P in "${POS_TH[@]}"; do
      for R in "${ROT_TH[@]}"; do

        jq --arg dm $DM --argjson le $LE --argjson p $P --argjson r $R '
            .delta_method = $dm
            | .L_eff = $le
            | .min_delta_pos = $p
            | .min_delta_rot = $r
        ' $BASE > $TMP

        FILE="${DM}_Le${LE}_P${P}_R${R}.log"
        echo "Running $FILE"

        python $SCRIPT \
            --pretrained_checkpoint "$CKPT" \
            --task_suite_name libero_object \
            --qk_config_json $TMP \
            > $OUT/$FILE 2>&1
      done
    done
  done
done
