from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
from mathruler.grader import extract_boxed_content, grade_answer
import os
import random

import torch.distributed as dist
import argparse  # 导入 argparse 用于解析命令行参数
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def parse_args():
    """解析从 shell 脚本传递的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Evaluate VLM models on specified datasets."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model being evaluated.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to evaluate on.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output results.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for inference."
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Whether activate COT through prompt engineering.",
    )
    return parser.parse_args()


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"Process {rank}/{world_size} initialized on cuda:{local_rank}")
    return local_rank, world_size, rank


def main():
    """主执行函数"""
    args = parse_args()
    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"

    # --- 根据传入的 dataset_name 配置数据集路径和模板 ---
    if rank == 0:
        print(f"Configuring for dataset: {args.dataset_name}")

    if args.dataset_name == "C-VQA-Real":
        if args.cot:
            format = " You FIRST think about the reasoning process step by step as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}."
        else:
            format = " Please directly give out the final answer, which MUST BE put in \\boxed{{}}."
        DATA_ROOT = "eval_data"
        IMAGE_ROOT = "/eaas/default/groups/xitucheng213/home/u2021213615/share/swr/Datasets/C-VQA/C-VQA-Real/C-VQA-Real_images"
        QUESTION_TEMPLATE = "{Question}" + format
        # QUESTION_TEMPLATE = "{Question} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
        ds_path = os.path.join(DATA_ROOT, "C-VQA-Real.json")

    elif args.dataset_name == "C-VQA-Synthetic":
        if args.cot:
            format = " You FIRST think about the reasoning process step by step as an internal monologue and then provide the final answer(For multiple-choice questions, please provide only the option letter, e.g., A.). The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}."
        else:
            format = " Please directly give out the final answer(For multiple-choice questions, please provide only the option letter, e.g., A.), which MUST BE put in \\boxed{{}}."
        DATA_ROOT = "eval_data"
        IMAGE_ROOT = "/eaas/default/groups/xitucheng213/home/u2021213615/share/swr/Datasets/C-VQA/C-VQA-Synthetic/C-VQA-Synthetic_images"
        # QUESTION_TEMPLATE = "{Question} Please directly give out the final answer. For multiple-choice questions, please provide only the option letter, e.g., A."
        QUESTION_TEMPLATE = "{Question}" + format
        ds_path = os.path.join(DATA_ROOT, "C-VQA-Synthetic.json")

    elif args.dataset_name == "MARS_Bench":
        if args.cot:
            format = " You FIRST think about the reasoning process step by step as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}."
        else:
            format = " Please directly give out the final answer, which MUST BE put in \\boxed{{}}."
        DATA_ROOT = "eval_data"
        IMAGE_ROOT = "/eaas/default/groups/xitucheng213/home/u2021213615/share/swr/Datasets/coco/val2014"
        QUESTION_TEMPLATE = "{Question}" + format
        ds_path = os.path.join(DATA_ROOT, "MARS_Bench.json")

    if rank == 0:
        print(f"Loading model: {args.model_name} from {args.model_path}")

    model = None
    processor = None

    # 使用 if/elif 来区分并加载不同模型
    if "qwen2.5-vl-3b" or "qwen2.5-vl-7b" in args.model_name.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": local_rank},
        )
        processor = AutoProcessor.from_pretrained(args.model_path)
    # 示例：可以这样添加对其他模型的支持
    # elif 'llava' in args.model_name.lower():
    #     model = LlavaForConditionalGeneration.from_pretrained(...)
    #     processor = AutoProcessor.from_pretrained(...)

    else:
        raise ValueError(f"Model '{args.model_name}' is not supported for loading.")

    if model is None or processor is None:
        raise RuntimeError("Model or processor failed to load.")

    # --- 准备数据和评估流程 ---
    if rank == 0:
        print(f"Processing {args.dataset_name}...")

    with open(ds_path, "r") as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)

    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    messages = []
    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x["image"])
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=x["problem"]),
                    },
                ],
            }
        ]
        messages.append(message)

    rank_outputs = []
    # 使用 args.batch_size
    for i in tqdm(range(0, len(messages), args.batch_size), disable=rank != 0):
        batch_messages = messages[i : i + args.batch_size]
        text = [
            processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in batch_messages
        ]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)

        generated_ids = model.generate(
            **inputs, use_cache=True, max_new_tokens=256, do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        rank_outputs.extend(batch_output_text)  # 简化输出，只保留文本结果

    print(f"Rank {rank} has finished processing {len(rank_outputs)} examples")

    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs)]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)

    if rank == 0:
        # 主进程收集所有结果
        for results in gathered_results:
            for idx, output in results:
                all_outputs[idx] = output

        final_output = []
        totals = defaultdict(int)
        corrects = defaultdict(int)
        all_q_type = []
        for i, (input_example, model_answer_raw) in enumerate(zip(data, all_outputs)):
            ground_truth = input_example["solution"]
            model_answer = model_answer_raw.strip()
            # is_correct = model_answer.casefold() == ground_truth.casefold()
            answer = extract_boxed_content(model_answer)
            if answer == "None":
                answer = model_answer
            is_correct = 1.0 if grade_answer(answer, ground_truth) else 0.0

            cf_type = "cf" if input_example["is_cf"] else "ncf"
            if (
                args.dataset_name == "C-VQA-Real"
                or args.dataset_name == "C-VQA-Synthetic"
            ):
                q_type = input_example.get("type", "unknown")
            else:
                # 其他数据集的处理
                q_type = input_example.get("type")
            if q_type not in all_q_type:
                all_q_type.append(q_type)
            totals["all"] += 1
            totals[cf_type] += 1
            totals[q_type] += 1

            if is_correct:
                corrects["all"] += 1
                corrects[cf_type] += 1
                corrects[q_type] += 1

            final_output.append(
                {
                    "image": input_example["image"],
                    "question": input_example["problem"],
                    "ground_truth": ground_truth,
                    "model_output": model_answer_raw,
                    "extracted_answer": answer,
                    "correct": 1 if is_correct else 0,
                }
            )

        accuracies = {}
        # metric_keys = ['all', 'ncf', 'cf', 'direct', 'indirect', 'boolean']
        metric_keys = list(totals.keys())

        print(f"\n--- Accuracy Report for {args.dataset_name} ---")
        for key in metric_keys:
            total_count = totals[key]
            correct_count = corrects[key]
            if total_count > 0:
                accuracy = (correct_count / total_count) * 100
                accuracies[f"{key}_accuracy"] = accuracy
                print(
                    f"{key.capitalize()} accuracy: {accuracy:.2f}% ({correct_count}/{total_count})"
                )
            else:
                accuracies[f"{key}_accuracy"] = 0
                print(f"{key.capitalize()} accuracy: N/A (0 samples)")

        # --- 动态生成输出文件名并保存 ---
        # 更清晰安全的写法
        suffix = "_COT_" if args.cot else ""
        output_filename = f"results_{args.model_name}_{args.dataset_name}{suffix}.json"

        output_path = os.path.join(args.output_dir, output_filename)
        os.makedirs(args.output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump({"accuracies": accuracies, "results": final_output}, f, indent=4)

        print(f"\nResults saved to {output_path}")
        print("-" * 100)

    dist.barrier()


if __name__ == "__main__":
    main()
