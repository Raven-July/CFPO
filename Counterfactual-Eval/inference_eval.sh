# source activate /eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/miniconda3/envs/EasyR1-031
cd /eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval
MODEL_NAME="Qwen2.5-VL-7B"
MODEL_PATH="/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Pretrained/Qwen2.5-VL-7B-Instruct"

DATASET_NAME="MARS_Bench"

OUTPUT_DIR="/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_result"
BATCH_SIZE=16
NUM_GPUS=1

echo "Starting evaluation for model: $MODEL_NAME on dataset: $DATASET_NAME"

torchrun --nproc_per_node=$NUM_GPUS inference_eval.py \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    # --cot

echo "Evaluation finished for $MODEL_NAME on $DATASET_NAME."