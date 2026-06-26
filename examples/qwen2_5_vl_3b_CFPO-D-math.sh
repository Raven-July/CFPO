#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

export SWANLAB_DIR='/Counterfact-Projects/train_logs'
export SWANLAB_MODE='local'

MODEL_PATH=/Pretrained/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

KL_CMVE_COEF=0.01

## Double Entropy Loss
USE_CMVE_ENTROPY_LOSS=false
CMVE_ENTROPY_LOSS_COEF=0.03
USE_ORI_ENTROPY_LOSS=true
ORI_ENTROPY_LOSS_COEF=0.03

cd /Counterfact-Projects/Counterfactual-R1

# 从 JSON 配置文件读取数据集配置
train_files=$(python3 - <<PY
import json
import os
with open("./examples/datasets_math_train.json") as f:
    configs = json.load(f)
print(json.dumps(configs, ensure_ascii=False))
PY
)

val_files=$(python3 - <<PY
import json
import os
with open("./examples/datasets_math_val.json") as f:
    configs = json.load(f)
print(json.dumps(configs, ensure_ascii=False))
PY
)

python3 -m verl.trainer.main \
    config=./examples/config.yaml \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.format_prompt=./examples/format_prompt/base.jinja \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.rollout_batch_size=384 \
    data.val_batch_size=512 \
    data.min_pixels=200704 \
    data.max_pixels=1003520 \
    data.mini_rollout_batch_size=128 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.enable_gradient_checkpointing=True \
    worker.actor.model.apply_cmve=True \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.rollout.n=5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.val_override_config='{"n":1, "temperature":0.0, "do_sample":False}' \
    worker.reward.reward_function=./examples/reward_function/base.py:compute_score \
    trainer.val_before_train=True \
    trainer.project_name=Counterfactual-R1 \
    trainer.experiment_name=qwen2_5_vl_3b_CFPO-D-math-bs384_V3_noref_orientropy_lowcoef \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=2 \
    trainer.val_generations_to_log=30 \
    trainer.total_epochs=2 \
    trainer.val_freq=5 \
    trainer.save_freq=50 \
    trainer.save_limit=5 \
    trainer.save_checkpoint_path=/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CFPO-D-math-bs384_V3_noref_orientropy_lowcoef \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=true \
    algorithm.adv_estimator=dapo \
    algorithm.kl_coef=0.0 \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    algorithm.use_kl_cmve=True \
    algorithm.kl_cmve_penalty=low_var_kl \
    algorithm.kl_cmve_schedule=fixed \
    algorithm.kl_cmve_coef=${KL_CMVE_COEF} \
    algorithm.use_cmve_entropy_loss=${USE_CMVE_ENTROPY_LOSS} \
    algorithm.cmve_entropy_loss_coef=${CMVE_ENTROPY_LOSS_COEF} \
    algorithm.use_ori_entropy_loss=${USE_ORI_ENTROPY_LOSS} \
    algorithm.ori_entropy_loss_coef=${ORI_ENTROPY_LOSS_COEF}
