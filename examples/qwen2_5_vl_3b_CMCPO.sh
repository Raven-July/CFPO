#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

export SWANLAB_DIR='/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/train_logs'
export SWANLAB_MODE='local'

BASE_PATH=/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy

MODEL_PATH=${BASE_PATH}/Pretrained/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

KL_CMVE_COEF=0.02

## Double Entropy Loss
USE_CMVE_ENTROPY_LOSS=false
CMVE_ENTROPY_LOSS_COEF=0.03
USE_ORI_ENTROPY_LOSS=false
ORI_ENTROPY_LOSS_COEF=0.03

cd ${BASE_PATH}/Counterfact-Projects/Counterfactual-R1

train_files='['
train_files="${train_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V3/A-OK-VQA_train.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/A-OK-VQA_images\"},"
train_files="${train_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V3/GQA_train.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/GQA_images\"},"
train_files="${train_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V3/vqacp_v2_train.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_trainval2014_images\"}" # <-- 最后一个没有逗号
train_files="${train_files}]"

val_files='['
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V3/A-OK-VQA_val_V3.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/A-OK-VQA_images\"},"
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V3/C-VQA-Real_val_V3.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images\"},"
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V3/MARS_Bench_val_V3.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images\"},"
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V3/GQA_val_V3.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/GQA_images\"},"
val_files="${val_files}{\"dataset\":\"${BASE_PATH}/Counterfact-Projects/Datasets/V3/vqacp_v2_val_V3.json\",\"image\":\"${BASE_PATH}/Counterfact-Projects/Datasets/COCO_trainval2014_images\"}" # <-- 最后一个没有逗号
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
    trainer.experiment_name=qwen2_5_vl_3b_CMCPO_V3_ref_noentropy \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=2 \
    trainer.val_generations_to_log=30 \
    trainer.total_epochs=1 \
    trainer.val_freq=5 \
    trainer.save_freq=50 \
    trainer.save_limit=4 \
    trainer.save_checkpoint_path=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CMCPO_V3_ref_noentropy \
    algorithm.use_kl_cmve=True \
    algorithm.kl_cmve_penalty=low_var_kl \
    algorithm.kl_cmve_schedule=fixed \
    algorithm.kl_cmve_coef=${KL_CMVE_COEF} \
    algorithm.use_cmve_entropy_loss=${USE_CMVE_ENTROPY_LOSS} \
    algorithm.cmve_entropy_loss_coef=${CMVE_ENTROPY_LOSS_COEF} \
    algorithm.use_ori_entropy_loss=${USE_ORI_ENTROPY_LOSS} \
    algorithm.ori_entropy_loss_coef=${ORI_ENTROPY_LOSS_COEF}
