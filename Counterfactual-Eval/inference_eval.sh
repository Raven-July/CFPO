cd /home/sunwanran/code/tmp/Counterfactual-R1/Counterfactual-Eval
MODEL_NAME="qwen2.5-vl-3b"
MODEL_PATH="/home/maxinyu/ckpt/Qwen2.5-VL-3B-Instruct"

DATASET_NAME="C-VQA-Synthetic"

OUTPUT_DIR="/home/sunwanran/code/tmp/Counterfactual-R1/Counterfactual-Eval/outputs"
BATCH_SIZE=4
NUM_GPUS=8

echo "Starting evaluation for model: $MODEL_NAME on dataset: $DATASET_NAME"

torchrun --nproc_per_node=$NUM_GPUS inference_eval.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE

echo "Evaluation finished for $MODEL_NAME on $DATASET_NAME."