# BASE_PATH=/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy
BASE_PATH=/home/u2021213615/share/yzy

cd ${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval
MODEL_NAME=Qwen2.5-VL-7B
# MODEL_PREFIX="MARS-fold1"
MODEL_PREFIX=""
MODEL_PATH=${BASE_PATH}/Pretrained/Qwen2.5-VL-7B-Instruct

DATASET_NAME=C-VQA-Synthetic

OUTPUT_DIR=${BASE_PATH}/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_result
BATCH_SIZE=4
NUM_GPUS=4

echo "Starting evaluation for model: $MODEL_NAME on dataset: $DATASET_NAME"

torchrun --nproc_per_node=$NUM_GPUS inference_eval.py \
    --model_name "$MODEL_NAME" \
    --model_prefix "$MODEL_PREFIX" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    # --cot
echo "Evaluation finished for $MODEL_NAME on $DATASET_NAME."