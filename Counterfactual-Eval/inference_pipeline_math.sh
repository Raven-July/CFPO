#!/bin/bash

BASE_PATH=/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy
cd ${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval

# --- 全局配置 ---
OUTPUT_DIR=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_result_math_8_grpo_exper

MODEL_NAME=Qwen2.5-VL-3B
MODEL_PREFIX="CMCPO-G-orientropy-math-202"
# MODEL_PREFIX="PAPO-D"
# MODEL_PATH=${BASE_PATH}/Pretrained/Qwen2.5-VL-3B-Instruct
# MODEL_PATH=${BASE_PATH}/Pretrained/PAPO-D-H-Qwen2.5-VL-3B
MODEL_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CMCPO-G-math-bs384_V3_ref_orientropy/global_step_202/actor/huggingface
# MODEL_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CMCPO-D-math-bs384_V3_noref_orientropy_lowcoef/global_step_202/actor/huggingface

# --- 配置文件路径 ---
# 注意：你需要确保这个文件包含所有数据集的配置
CONFIG_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/datasets_config_math.json

# --- 替换配置文件中的 BASE_PATH 变量 ---
# 由于 JSON 不支持 shell 变量，我们在这里进行替换
TEMP_CONFIG_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/temp_datasets_config.json
sed "s|\${BASE_PATH}|$BASE_PATH|g" "${CONFIG_PATH}" > "${TEMP_CONFIG_PATH}"

echo "Starting batch evaluation for model: $MODEL_NAME on all datasets defined in $CONFIG_PATH"

python inference_eval.py \
    --model_name "$MODEL_NAME" \
    --model_prefix "$MODEL_PREFIX" \
    --model_path "$MODEL_PATH" \
    --config_path "$TEMP_CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n 8 \
    --cot  

# --- 清理临时文件 ---
rm "${TEMP_CONFIG_PATH}"

echo "Batch evaluation finished for $MODEL_NAME."