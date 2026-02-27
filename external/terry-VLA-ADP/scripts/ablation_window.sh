#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
CKPT="checkpoints/openvla-7b-oft-finetuned-libero-spatial"
TASK=libero_spatial
BASE_CONFIG=experiments/robot/libero/configs/ablation_windows/base_prune.json
LOG_DIR=logs/spatial_logs
TMP_CONFIG=tmp_prune_window.json

mkdir -p ${LOG_DIR}

for W in 4 5 6 7 8 9 10
do
    echo "=============================="
    echo " Running adjacent_extrema_window = ${W}"
    echo "=============================="


    jq --argjson w ${W} '.adjacent_extrema_window = $w' \
        ${BASE_CONFIG} > ${TMP_CONFIG}

    python ${SCRIPT} \
        --pretrained_checkpoint "${CKPT}" \
        --task_suite_name ${TASK} \
        --qk_config_json ${TMP_CONFIG} \
        > ${LOG_DIR}/LIBERO_Spatial_Ablation_w${W}.log 2>&1
done


rm -f ${TMP_CONFIG}
