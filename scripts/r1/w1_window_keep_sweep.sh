#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
CKPT="checkpoints/openvla-7b-oft-finetuned-libero-spatial"
BASE=experiments/robot/libero/configs/base_prune.json
TMP=tmp_sweep.json

OUT=logs/r1_window_keep
mkdir -p $OUT

WINDOWS=(3 5 7 9)
KEEPS=(1.0 0.75 0.5 0.35)

for W in "${WINDOWS[@]}"; do
    for KR in "${KEEPS[@]}"; do
        echo "window=$W, keep=$KR"

        jq --argjson w $W --argjson kr $KR '
            .adjacent_extrema_window = $w
            | .qk_keep_ratio = $kr
            | .use_dynamic_visual_strategy = true
        ' $BASE > $TMP

        python $SCRIPT \
            --pretrained_checkpoint "$CKPT" \
            --task_suite_name libero_spatial \
            --qk_config_json $TMP \
            > $OUT/w${W}_kr${KR}.log 2>&1
    done
done
