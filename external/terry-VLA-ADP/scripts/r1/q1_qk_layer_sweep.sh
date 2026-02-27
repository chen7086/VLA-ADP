#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
CKPT="checkpoints/openvla-7b-oft-finetuned-libero-spatial"
BASE=experiments/robot/libero/configs/base_prune.json
TMP=tmp_qk.json

OUT=logs/r1_qk_layers
mkdir -p $OUT

LAYERS=(8 16 24 31)

for L in "${LAYERS[@]}"; do
    echo "qk_layer=$L"

    jq --argjson L $L '
        .qk_layer = $L
    ' $BASE > $TMP

    python $SCRIPT \
        --pretrained_checkpoint "$CKPT" \
        --task_suite_name libero_spatial \
        --qk_config_json $TMP \
        > $OUT/layer_${L}.log 2>&1
done
