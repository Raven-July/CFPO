#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

export SWANLAB_DIR='/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/train_logs'
export SWANLAB_MODE='local'

BASE_PATH=/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy

MODEL_PATH=${BASE_PATH}/Pretrained/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

cd ${BASE_PATH}/Counterfact-Projects/Counterfactual-R1

train_files='['
train_files="${train_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/A-OK-VQA_train_V2.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/A-OK-VQA_images\"},"
train_files="${train_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/C-VQA-Real_train_V2.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images\"},"
train_files="${train_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/C-VQA-Synthetic_train_V2.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/C-VQA-Synthetic_images\"},"
train_files="${train_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/MARS_Bench_train_V2.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images\"},"
train_files="${train_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/ViRL39K_train_fixed.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/ViRL39K_images\"},"
train_files="${train_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/vqacp_v2_train_V2.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_trainval2014_images\"}" # <-- 最后一个没有逗号
train_files="${train_files}]"

val_files='['
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/A-OK-VQA_val_V2.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/A-OK-VQA_images\"},"
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/C-VQA-Real_val_V2.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images\"},"
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/geometry3k_val.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/geometry3k_images\"},"
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/MARS_Bench_val_V2.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images\"},"
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/ViRL39K_val_fixed.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/ViRL39K_images\"},"
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V2/vqacp_v2_val_V2.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_trainval2014_images\"}" # <-- 最后一个没有逗号
val_files="${val_files}]"

python3 -m verl.trainer.main \
    config=./examples/config.yaml \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.format_prompt=./examples/format_prompt/base.jinja \
    data.max_prompt_length=4096 \
    data.rollout_batch_size=128 \
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
    worker.actor.optim.lr=2e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.rollout.n=5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.val_override_config='{"n":1, "temperature":0.0, "do_sample":False}' \
    worker.reward.reward_function=./examples/reward_function/base.py:compute_score \
    trainer.val_before_train=False \
    trainer.project_name=Counterfactual-R1 \
    trainer.experiment_name=qwen2_5_vl_3b_CMCPO_V2_GRPO_fixed \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=2 \
    trainer.val_generations_to_log=30 \
    trainer.total_epochs=2 \
    trainer.val_freq=5 \
    trainer.save_freq=32 \
    trainer.save_limit=3 \
    trainer.save_checkpoint_path=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CMCPO_V2_GRPO_fixed \
    algorithm.use_kl_cmve=False \
