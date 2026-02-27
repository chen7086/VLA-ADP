#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

SCRIPT=experiments/robot/libero/run_libero_eval_prune_v2.py
CKPT="checkpoints/openvla-7b-oft-finetuned-libero-spatial"
TASK=libero_spatial

BASE_CONFIG=experiments/robot/libero/configs/ablation_layers/base_prune_layer.json
TMP_CONFIG=tmp_prune_layer.json
LOG_DIR=logs/spatial_logs

mkdir -p ${LOG_DIR}

for L in 8 16 24 31
do
    echo "=============================="
    echo " Running qk_layer = ${L}"
    echo "=============================="

    # jq --argjson l ${L} '.qk_layer = [$l]' \
    #     ${BASE_CONFIG} > ${TMP_CONFIG}

    jq --argjson l ${L} '.qk_layer = $l' \
        ${BASE_CONFIG} > ${TMP_CONFIG}

    python ${SCRIPT} \
        --pretrained_checkpoint "${CKPT}" \
        --task_suite_name ${TASK} \
        --qk_config_json ${TMP_CONFIG} \
        > ${LOG_DIR}/LIBERO_Spatial_Ablation_Layer${L}.log 2>&1
done

rm -f ${TMP_CONFIG}
