BASE_PATH=/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy
# BASE_PATH=/home/u2021213615/share/yzy

cd ${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval
MODEL_NAME=Qwen2.5-VL-3B
# MODEL_PREFIX="cmve-a1-MARS-fold1"
MODEL_PREFIX="CMCPO-fixed"
# MODEL_PATH=${BASE_PATH}/Pretrained/Qwen2.5-VL-3B-Instruct
# MODEL_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CMCPO_V2_GRPO/global_step_160/actor/huggingface
MODEL_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/checkpoints/qwen2_5_vl_3b_CMCPO_V2_ref_noentropy_exper2/global_step_160/actor/huggingface

# JSON_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_data/C-VQA-Real_test_V2.json
# JSON_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_data/MARS_Bench_test_V2.json
# IMAGE_ROOT=${BASE_PATH}/Counterfact-Projects/Datasets/COCO_val2014_images

# JSON_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_data/C-VQA-Synthetic_test_V2.json
# IMAGE_ROOT=${BASE_PATH}/Counterfact-Projects/Datasets/C-VQA-Synthetic_images

# JSON_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_data/geometry3k_full.json
# IMAGE_ROOT=${BASE_PATH}/Counterfact-Projects/Datasets/geometry3k_images

# JSON_PATH=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_data/A-OK-VQA_val.json
# IMAGE_ROOT=${BASE_PATH}/Counterfact-Projects/Datasets/A-OK-VQA_images

JSON_PATH=${BASE_PATH}/Counterfact-Projects/Datasets/V2/ViRL39K_test_V2.json
# JSON_PATH=${BASE_PATH}/Counterfact-Projects/Datasets/V2/ViRL39K_test_fixed.json
IMAGE_ROOT=${BASE_PATH}/Counterfact-Projects/Datasets/ViRL39K_images

OUTPUT_DIR=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_result
BATCH_SIZE=64
NUM_GPUS=2

echo "Starting evaluation for model: $MODEL_NAME on dataset: $DATASET_NAME"

python inference_eval.py \
    --model_name "$MODEL_NAME" \
    --model_prefix "$MODEL_PREFIX" \
    --model_path "$MODEL_PATH" \
    --json_path "$JSON_PATH" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --cot
echo "Evaluation finished for $MODEL_NAME on $DATASET_NAME."