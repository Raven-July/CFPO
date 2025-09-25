import os
import re
import json
import math
import random
import argparse
import warnings
from collections import defaultdict
from typing import Optional
from PIL import Image

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from mathruler.grader import grade_answer

# ----------------- 全局配置 -----------------
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
MIN_PIXELS = 262_144  # 最小像素数
MAX_PIXELS = 4_194_304  # 最大像素数


# ===================== 参数解析 =====================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Evaluate VLM models on specified datasets."
    )
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--model_prefix", type=str, default="", help="模型前缀（可选）")
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="要评估的数据集名称"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="保存结果的目录")
    parser.add_argument("--batch_size", type=int, default=4, help="推理批大小")
    parser.add_argument(
        "--cot", action="store_true", help="是否启用 CoT（链式思维）推理"
    )
    return parser.parse_args()


# ===================== 分布式设置 =====================
def setup_distributed():
    """初始化分布式训练/推理环境"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    print(f"[Distributed Init] Rank {rank}/{world_size} on cuda:{local_rank}")
    return local_rank, world_size, rank


# ===================== 图像预处理 =====================
def process_image(
    image: Image.Image, min_pixels: Optional[int], max_pixels: Optional[int]
) -> Image.Image:
    """根据最小/最大像素约束调整图像大小，并转换为 RGB"""
    image.load()  # 避免 "Too many open files"

    # 缩小到最大像素
    if max_pixels and image.width * image.height > max_pixels:
        factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))

    # 放大到最小像素
    if min_pixels and image.width * image.height < min_pixels:
        factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))

    # 转换为 RGB 模式
    return image.convert("RGB") if image.mode != "RGB" else image


# ===================== 数据加载 =====================
def get_dataset_config(dataset_name: str):
    """根据数据集名称返回路径和图片根目录"""
    DATA_ROOT = "eval_data"
    if dataset_name == "C-VQA-Real":
        return (
            os.path.join(DATA_ROOT, "C-VQA-Real.json"),
            "/eaas/default/groups/xitucheng213/home/share/swr/Datasets/Counterfactual/C-VQA/C-VQA-Real/C-VQA-Real_images",
        )
    elif dataset_name == "C-VQA-Synthetic":
        return (
            os.path.join(DATA_ROOT, "C-VQA-Synthetic.json"),
            "/eaas/default/groups/xitucheng213/home/share/swr/Datasets/Counterfactual/C-VQA/C-VQA-Synthetic/C-VQA-Synthetic_images",
        )
    elif dataset_name == "MARS_Bench":
        return (
            os.path.join(DATA_ROOT, "MARS_Bench.json"),
            "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/COCO_val2014",
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


# ===================== 模型加载 =====================
def load_model_and_processor(model_name: str, model_path: str, device):
    """根据名称加载模型和处理器"""
    if "qwen2.5-vl-3b" in model_name.lower() or "qwen2.5-vl-7b" in model_name.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return model, processor
    else:
        raise ValueError(f"暂不支持加载模型: {model_name}")


# ===================== 主流程 =====================
def main():
    args = parse_args()
    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"

    # ---- 数据集与 Prompt 配置 ----
    ds_path, image_root = get_dataset_config(args.dataset_name)
    prompt_template = (
        "The user asks a question, and then you solve it. "
        "You FIRST think about the reasoning process step by step as an internal monologue "
        "and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer MUST BE enclosed within <answer> </answer> tags."
        if args.cot
        else "The user asks a question, and then you solve it. "
        "Please directly give out the final answer, which MUST BE enclosed within <answer> </answer> tags."
    )

    if rank == 0:
        print(f"[INFO] Loading dataset: {args.dataset_name}")
        print(f"[INFO] Loading model: {args.model_name} from {args.model_path}")

    model, processor = load_model_and_processor(
        args.model_name, args.model_path, local_rank
    )

    # ---- 数据加载与划分 ----
    with open(ds_path, "r") as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)
    per_rank_data = len(data) // world_size
    start_idx, end_idx = rank * per_rank_data, (
        (rank + 1) * per_rank_data if rank < world_size - 1 else len(data)
    )
    rank_data = data[start_idx:end_idx]

    # ---- 构建消息 ----
    messages = []
    for item in rank_data:
        image_path = os.path.join(image_root, item["image"])
        messages.append(
            [
                {"role": "system", "content": prompt_template},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": item["problem"]},
                    ],
                },
            ]
        )

    # ---- 批量推理 ----
    outputs = []
    for i in tqdm(range(0, len(messages), args.batch_size), disable=rank != 0):
        batch_messages = messages[i : i + args.batch_size]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        resized_images = (
            [process_image(img, MIN_PIXELS, MAX_PIXELS) for img in image_inputs]
            if image_inputs
            else None
        )

        inputs = processor(
            text=texts,
            images=resized_images,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)

        generated_ids = model.generate(
            **inputs, use_cache=True, max_new_tokens=256, do_sample=False
        )
        trimmed_ids = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        outputs.extend(processor.batch_decode(trimmed_ids, skip_special_tokens=True))

    print(f"[INFO] Rank {rank} 完成 {len(outputs)} 条样本处理")

    # ---- 分布式收集结果 ----
    gathered_results = [None] * world_size
    dist.all_gather_object(
        gathered_results, [(start_idx + i, o) for i, o in enumerate(outputs)]
    )

    # ---- 主进程计算准确率并保存 ----
    if rank == 0:
        all_outputs = [None] * len(data)
        for results in gathered_results:
            for idx, output in results:
                all_outputs[idx] = output

        totals, corrects, final_output = defaultdict(int), defaultdict(int), []
        for i, (item, raw_output) in enumerate(zip(data, all_outputs)):
            gt = item["solution"]
            match = re.search(
                r"<answer>(.*?)</answer>", raw_output, re.DOTALL
            ) or re.search(r"<answer>(.*)", raw_output, re.DOTALL)
            answer = match.group(1).strip() if match else raw_output.strip()
            is_correct = grade_answer(answer, gt)

            cf_type = "cf" if item.get("is_cf") else "ncf"
            q_type = item.get("type", "unknown")

            # 统计数量
            for key in ["all", cf_type, q_type, f"{q_type}_{cf_type}"]:
                totals[key] += 1
                if is_correct:
                    corrects[key] += 1

            final_output.append(
                {
                    "image": item["image"],
                    "question": item["problem"],
                    "is_cf": item.get("is_cf", False),
                    "type": item.get("type", "unknown"),
                    "ground_truth": gt,
                    "model_output": raw_output,
                    "extracted_answer": answer,
                    "correct": int(is_correct),
                }
            )

        # ---- 计算准确率 ----
        print(f"\n--- {args.dataset_name} 准确率报告 ---")
        accuracies = {}
        for key in sorted(totals.keys()):
            acc = (corrects[key] / totals[key]) * 100 if totals[key] > 0 else 0
            accuracies[f"{key}_accuracy"] = acc
            print(f"{key:<12} : {acc:.2f}% ({corrects[key]}/{totals[key]})")

        # ---- 保存结果 ----
        suffix = "_COT_" if args.cot else ""
        prefix = f"-{args.model_prefix}" if args.model_prefix else ""
        filename = f"results_{args.model_name}{prefix}_{args.dataset_name}{suffix}.json"
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, filename)

        with open(output_path, "w") as f:
            json.dump({"accuracies": accuracies, "results": final_output}, f, indent=4)

        print(f"\n[INFO] 结果已保存到: {output_path}\n{'-' * 100}")

    dist.barrier()


if __name__ == "__main__":
    main()
