#!/bin/bash
cd Counterfactual-R1/Counterfactual-Eval

# --- 全局配置 ---
OUTPUT_DIR=Counterfactual-R1/Counterfactual-Eval/eval_result

MODEL_NAME=Qwen2.5-VL-3B
MODEL_PREFIX="CFPO-G-ref-noentropy-math-202"
MODEL_PATH=Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CFPO-G-ref-noentropy-math-202/global_step_202/actor/huggingface

# --- 配置文件路径 ---
# 注意：你需要确保这个文件包含所有数据集的配置
CONFIG_PATH=Counterfactual-R1/Counterfactual-Eval/datasets_config_math.json

echo "Starting batch evaluation for model: $MODEL_NAME on all datasets defined in $CONFIG_PATH"

python inference_eval.py \
    --model_name "$MODEL_NAME" \
    --model_prefix "$MODEL_PREFIX" \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n 8 \
    --cot

echo "Batch evaluation finished for $MODEL_NAME."