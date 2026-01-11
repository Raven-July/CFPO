#!/bin/bash

BASE_PATH=/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy
cd ${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval

# --- 全局配置 ---
OUTPUT_DIR=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_result_math_8_dapo

MODEL_NAME=Qwen2.5-VL-3B
MODEL_PREFIX="CMCPO-D-lowcoef-math-202"
# MODEL_PREFIX="PAPO"
# MODEL_PATH=${BASE_PATH}/Pretrained/Qwen2.5-VL-3B-Instruct
# MODEL_PATH=${BASE_PATH}/Pretrained/PAPO-G-H-Qwen2.5-VL-3B
MODEL_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CMCPO-D-math-bs384_V3_noref_noentropy_lowcoef/global_step_202/actor/huggingface
# MODEL_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CMCPO_V3_ref_noentropy/global_step_304/actor/huggingface

# --- 配置文件路径 ---
# 注意：你需要确保这个文件包含所有数据集的配置
CONFIG_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/datasets_config.json

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

    # {
    #     "dataset_name": "geometry3k",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3-math/geometry3k_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/geometry3k_images"
    # },
    # {
    #     "dataset_name": "MathVista",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3-math/MathVista_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/MathVista_images"
    # },
    # {
    #     "dataset_name": "WeMath",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3-math/WeMath_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/WeMath_images"
    # },
    # {
    #     "dataset_name": "MMK12",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3-math/MMK12_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/MMK12_images"
    # },
    # {
    #     "dataset_name": "MathVerse_fixed",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3-math/MathVerse-closed_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/MathVerse_images"
    # },
    # {
    #     "dataset_name": "LogicVista",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3-math/LogicVista_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/LogicVista_images"
    # },
    # {
    #     "dataset_name": "Count",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3-math/Count_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/Count_images"
    # },
    # {
    #     "dataset_name": "MMMUPro-Vision",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3-math/MMMUPro-Vision_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/MMMUPro-Vision_images"
    # },
    # {
    #     "dataset_name": "POPE",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3-math/POPE_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images"
    # },
    # {
    #     "dataset_name": "A-OK-VQA",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3/A-OK-VQA_val.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/A-OK-VQA_images"
    # },
    # {
    #     "dataset_name": "MARS_Bench",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3/MARS_Bench_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images"
    # },
    # {
    #     "dataset_name": "C-VQA-Real",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3/C-VQA-Real_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images"
    # },
    # {
    #     "dataset_name": "C-VQA-Synthetic",
    #     "json_path": "${BASE_PATH}/Counterfact-Projects/Datasets/V3/C-VQA-Synthetic_test.json",
    #     "image_root": "${BASE_PATH}/Counterfact-Projects/Datasets/C-VQA-Synthetic_images"
    # }