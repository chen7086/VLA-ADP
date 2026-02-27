#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
CKPT="checkpoints/openvla-7b-oft-finetuned-libero-spatial"
TASK=libero_spatial
BASE=experiments/robot/libero/configs/base_prune.json
TMP=tmp_vis.json

# ---- Output dirs ----
OUT_STATIC=vis_Q2_static
OUT_ADP=vis_Q2_adp
mkdir -p $OUT_STATIC
mkdir -p $OUT_ADP

echo "================== Q2: Visualization =================="

###############################################
# 1. Baseline Static Pruning
###############################################
echo "----- Generating Baseline Static Visualization -----"

jq '
    .dump_episode = true
    | .save_videos = true
    | .overlay_pruned = true
    | .overlay_actions = true
    | .use_dynamic_visual_strategy = false
    | .dynamic_ablation_enabled = false
    | .dump_dir = "vis_Q2_static"
    | .num_trials_per_task = 2
' $BASE > $TMP

python $SCRIPT \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name $TASK \
    --qk_config_json $TMP



###############################################
# 2. Ours: ADP dynamic pruning
###############################################
echo "----- Generating ADP Visualization -----"

jq '
    .dump_episode = true
    | .save_videos = true
    | .overlay_pruned = true
    | .overlay_actions = true
    | .use_dynamic_visual_strategy = true
    | .dynamic_ablation_enabled = false
    | .dump_dir = "vis_Q2_adp"
    | .num_trials_per_task = 2
' $BASE > $TMP

python $SCRIPT \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name $TASK \
    --qk_config_json $TMP

echo "================== Q2 DONE =================="
