#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
CKPT="checkpoints/openvla-7b-oft-finetuned-libero-spatial"
TASKS=(libero_spatial libero_object)
BASE=experiments/robot/libero/configs/base_prune.json
TMP=tmp_q3.json

ABL_DIR=logs/Q3_gating_modes
TH_DIR=logs/Q3_threshold_sweep

mkdir -p $ABL_DIR
mkdir -p $TH_DIR

echo "================== Q3: Generalization Tests =================="


############################################################
# 1. Gating Mode Ablation (ours / always coarse / always fine)
############################################################
echo "----- Running Gating Mode Ablation -----"

for TASK in "${TASKS[@]}"; do

    ################ Ours ################
    jq '
        .use_dynamic_visual_strategy = true
        | .dynamic_ablation_enabled = false
    ' $BASE > $TMP

    python $SCRIPT \
        --pretrained_checkpoint "$CKPT" \
        --task_suite_name $TASK \
        --qk_config_json $TMP \
        > $ABL_DIR/${TASK}_ours.log 2>&1

    ################ Always Coarse ################
    jq '
        .use_dynamic_visual_strategy = false
        | .dynamic_ablation_enabled = true
        | .initial_state = 0
    ' $BASE > $TMP

    python $SCRIPT \
        --pretrained_checkpoint "$CKPT" \
        --task_suite_name $TASK \
        --qk_config_json $TMP \
        > $ABL_DIR/${TASK}_always_coarse.log 2>&1

    ################ Always Fine ################
    jq '
        .use_dynamic_visual_strategy = false
        | .dynamic_ablation_enabled = true
        | .initial_state = 1
    ' $BASE > $TMP

    python $SCRIPT \
        --pretrained_checkpoint "$CKPT" \
        --task_suite_name $TASK \
        --qk_config_json $TMP \
        > $ABL_DIR/${TASK}_always_fine.log 2>&1

done



############################################################
# 2. Threshold Robustness Sweep
############################################################
echo "----- Running Threshold Robustness Sweep -----"

L_EFF_VALUES=(0.10 0.15 0.20)
ROT_VALUES=(0.0 0.02)

for LE in "${L_EFF_VALUES[@]}"; do
    for MDR in "${ROT_VALUES[@]}"; do

        echo "Sweeping L_eff=${LE}, min_delta_rot=${MDR}"

        jq --argjson le $LE --argjson mdr $MDR '
            .use_dynamic_visual_strategy = true
            | .L_eff = $le
            | .min_delta_rot = $mdr
        ' $BASE > $TMP

        python $SCRIPT \
            --pretrained_checkpoint "$CKPT" \
            --task_suite_name libero_spatial \
            --qk_config_json $TMP \
            > $TH_DIR/Le_${LE}_Rot_${MDR}.log 2>&1

    done
done

echo "================== Q3 DONE =================="
