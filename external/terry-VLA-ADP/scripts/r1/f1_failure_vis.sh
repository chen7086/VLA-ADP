#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
CKPT="checkpoints/openvla-7b-oft-finetuned-libero-object"
BASE=experiments/robot/libero/configs/base_prune.json
TMP=tmp_fail.json

OUT=vis_failcases
mkdir -p $OUT

jq '
    .dump_episode = true
    | .save_videos = true
    | .overlay_pruned = true
    | .overlay_actions = true
    | .dump_dir = "vis_failcases"
    | .num_trials_per_task = 50
    | .use_dynamic_visual_strategy = true
' $BASE > $TMP

python $SCRIPT \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name libero_object \
    --qk_config_json $TMP
