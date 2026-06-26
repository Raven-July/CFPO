#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

export SWANLAB_DIR='/Counterfact-Projects/train_logs'
export SWANLAB_MODE='local'

MODEL_PATH=/Pretrained/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

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
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.enable_gradient_checkpointing=True \
    worker.actor.model.apply_cmve=False \
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
    trainer.experiment_name=qwen2_5_vl_3b_GRPO-math-bs384 \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=2 \
    trainer.val_generations_to_log=30 \
    trainer.total_epochs=2 \
    trainer.val_freq=5 \
    trainer.save_freq=50 \
    trainer.save_limit=5 \
    trainer.save_checkpoint_path=/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_GRPO-math-bs384 \
    algorithm.use_kl_cmve=False \
