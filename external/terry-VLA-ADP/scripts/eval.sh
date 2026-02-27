export CUDA_VISIBLE_DEVICES=1
python experiments/robot/libero/run_libero_eval_prune_v2.py \
  --pretrained_checkpoint "checkpoints/openvla-7b-oft-finetuned-libero-spatial" \
  --task_suite_name libero_spatial \
  --qk_config_json experiments/robot/libero/configs/prune_v4_config.json > logs/spatial_logs/LIBERO_Spatial_P_V4.log 2>&1