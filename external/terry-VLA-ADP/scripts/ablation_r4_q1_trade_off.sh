#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
CKPT="checkpoints/openvla-7b-oft-finetuned-libero-spatial"
TASK=libero_spatial
BASE=experiments/robot/libero/configs/base_prune.json
TMP=tmp_tradeoff.json

# ------------ Params ------------
KEEP_RATIOS=(1.0 0.75 0.5 0.35 0.25)

# ------------ Output dirs ------------
STATIC_DIR=logs/Q1_tradeoff_static
ADP_DIR=logs/Q1_tradeoff_adp

mkdir -p $STATIC_DIR
mkdir -p $ADP_DIR

echo "================== Q1: Token Reduction Trade-off =================="


###############################################
# 1. Static pruning (no dynamic gating)
###############################################
echo "----- Running STATIC pruning sweep -----"
for KR in "${KEEP_RATIOS[@]}"; do

    echo "[Static] qk_keep_ratio = ${KR}"

    jq --argjson kr $KR '
        .qk_keep_ratio = $kr
        | .qk_keep_enabled = true
        | .use_dynamic_visual_strategy = false
        | .dynamic_ablation_enabled = false
    ' $BASE > $TMP

    python $SCRIPT \
        --pretrained_checkpoint "$CKPT" \
        --task_suite_name $TASK \
        --qk_config_json $TMP \
        > $STATIC_DIR/kr_${KR}.log 2>&1

done



###############################################
# 2. ADP dynamic pruning
###############################################
echo "----- Running ADP pruning sweep -----"
for KR in "${KEEP_RATIOS[@]}"; do

    echo "[ADP] qk_keep_ratio = ${KR}"

    jq --argjson kr $KR '
        .qk_keep_ratio = $kr
        | .qk_keep_enabled = true
        | .use_dynamic_visual_strategy = true
        | .dynamic_ablation_enabled = false
    ' $BASE > $TMP

    python $SCRIPT \
        --pretrained_checkpoint "$CKPT" \
        --task_suite_name $TASK \
        --qk_config_json $TMP \
        > $ADP_DIR/kr_${KR}.log 2>&1

done

echo "================== Q1 DONE =================="
