#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

export SWANLAB_DIR='/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/train_logs'
export SWANLAB_MODE='local'

BASE_PATH=/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy

MODEL_PATH=${BASE_PATH}/Pretrained/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

cd ${BASE_PATH}/Counterfact-Projects/Counterfactual-R1

python3 -m verl.trainer.main \
    config=./examples/config.yaml \
    data.image_dir=${BASE_PATH}/Counterfact-Projects/Datasets/CMCPO_V1 \
    data.train_files=${BASE_PATH}/Counterfact-Projects/Datasets/CMCPO_train_V1.json \
    data.val_files=${BASE_PATH}/Counterfact-Projects/Datasets/MARS_Bench_test.json \
    data.format_prompt=./examples/format_prompt/base.jinja \
    data.max_prompt_length=4096 \
    data.rollout_batch_size=128 \
    data.val_batch_size=512 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.enable_gradient_checkpointing=True \
    worker.actor.model.apply_cmve=False \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=5e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.rollout.n=5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.val_override_config='{"n":1,"temperature":0.5}' \
    worker.reward.reward_function=./examples/reward_function/base.py:compute_score \
    trainer.val_before_train=True \
    trainer.project_name=Counterfactual-R1 \
    trainer.experiment_name=qwen2_5_vl_3b_GRPO_V1_New \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=2 \
    trainer.val_generations_to_log=20 \
    trainer.total_epochs=3 \
    trainer.val_freq=5 \
    trainer.save_freq=30 \
    trainer.save_limit=3 \
    trainer.save_checkpoint_path=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_GRPO_V1 \
