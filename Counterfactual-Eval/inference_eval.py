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

from tqdm import tqdm
from transformers import AutoProcessor
from mathruler.grader import grade_answer
from vllm import LLM, SamplingParams  # (vLLM) 导入 vLLM

# ----------------- 全局配置 -----------------
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
MIN_PIXELS = 200704  # 最小像素数
MAX_PIXELS = 1003520  # 最大像素数

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


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
        "--json_path", type=str, required=True, help="评估 JSON 文件的路径"
    )
    parser.add_argument(
        "--image_root", type=str, required=True, help="图像文件的根目录"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="保存结果的目录")
    # (vLLM) batch_size 不再由我们控制，但可以保留参数以兼容旧命令（尽管未使用）
    parser.add_argument("--batch_size", type=int, default=4, help="（vLLM 下未使用）")
    parser.add_argument(
        "--cot", action="store_true", help="是否启用 CoT（链式思维）推理"
    )
    # (vLLM) 添加 tensor_parallel_size 参数
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="vLLM 使用的 GPU 数量 (默认: all available)",
    )
    return parser.parse_args()


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

    if image.mode != "RGB":
        if image.mode == "P":
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")
    # 转换为 RGB 模式
    return image


# ===================== 主流程 =====================
def main():
    args = parse_args()
    # (vLLM) vLLM 自动处理多 GPU
    if args.tensor_parallel_size:
        world_size = args.tensor_parallel_size
    else:
        world_size = torch.cuda.device_count()

    # ---- 数据集与 Prompt 配置 ----
    ds_path = args.json_path
    image_root = args.image_root
    dataset_name = os.path.basename(ds_path).split(".")[0]

    prompt_template = (
        "The user asks a question, and then you solve it based on the images provided. "
        "You FIRST think about the reasoning process step by step as an internal monologue "
        "and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer(SIMPLEST WORDS) MUST BE enclosed within <answer> </answer> tags."
        if args.cot
        else "The user asks a question, and then you solve it based on the images provided. "
        "Please directly give out the final answer(SIMPLEST WORDS), which MUST BE enclosed within <answer> </answer> tags."
    )

    print(f"[INFO] Loading dataset from: {args.json_path}")
    print(
        f"[INFO] Loading model: {args.model_name} from {args.model_path} with TP={world_size}"
    )
    # (vLLM) 加载 vLLM 模型
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=world_size,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",  # 假设为 bfloat16
    )
    # (vLLM) 仍然需要 Processor 来构建提示模板
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # ---- 数据加载与划分 ----
    with open(ds_path, "r") as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)

    # (vLLM) vLLM 只需要一个统一的输入列表
    vllm_inputs = []

    print(f"[INFO] Preparing {len(data)} samples for vLLM...")
    # (vLLM) 循环 'data'
    for item in tqdm(data):
        problem_text = item["problem"]
        image_files = item["images"]

        # 1. 为这个 item 加载 PIL 图像
        item_images_pil = []
        for img_name in image_files:
            image_path = os.path.join(image_root, img_name)
            try:
                img = Image.open(image_path)
                img = process_image(img, MIN_PIXELS, MAX_PIXELS)
                item_images_pil.append(img)
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                # 添加一个占位符灰色图像以避免崩溃
                item_images_pil.append(Image.new("RGB", (448, 448), "grey"))

        # 2. 构建 'messages' 列表（使用虚拟图像路径）
        # Processor 需要这个结构来正确插入 <image> 相关的特殊 token
        content_list = []
        parts = problem_text.split("<image>")
        # 使用虚拟迭代器检查图像数量是否匹配
        image_iter_check = iter(item_images_pil)

        for i, part in enumerate(parts):
            if i != 0:
                try:
                    # 仅用于模板构建，路径不重要，但类型必须是 "image"
                    # (修改) 移除虚拟路径，仅保留类型
                    content_list.append({"type": "image"})
                    next(image_iter_check)  # 推进迭代器
                except StopIteration:
                    print(
                        f"Warning: Mismatch between <image> tags and image list for item id {item.get('id', 'N/A')}."
                    )

            if part:
                content_list.append({"type": "text", "text": part})

        messages_for_item = [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": content_list},
        ]

        # 3. 应用聊天模板获取最终的文本提示
        try:
            final_prompt_text = processor.apply_chat_template(
                messages_for_item, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"Error applying template for item {item.get('id', 'N/A')}: {e}")
            final_prompt_text = "Error: Could not format prompt."  # 占位符

        # 4. (修改) 存储 vLLM 需要的统一输入字典
        vllm_inputs.append(
            {
                "prompt": final_prompt_text,
                "multi_modal_data": {"image": item_images_pil},
            }
        )

    # ---- 批量推理 (vLLM) ----
    print("[INFO] Starting vLLM generation...")

    # (vLLM) 定义采样参数
    sampling_params = SamplingParams(
        temperature=0.0,  # 对应 do_sample=False
        max_tokens=2048,  # 对应 max_new_tokens
    )

    # (vLLM) 一次性调用 generate
    # (修改)
    vllm_outputs = llm.generate(
        prompts=vllm_inputs,  # 传入统一的输入列表
        sampling_params=sampling_params,
        # multi_modal_data=... # 已合并到 prompts 中
    )

    print(f"[INFO] vLLM generation complete. Received {len(vllm_outputs)} outputs.")

    # (vLLM) 从 vLLM 的 RequestOutput 对象中提取纯文本
    # vLLM 保证输出顺序与输入顺序一致
    all_outputs = [out.outputs[0].text for out in vllm_outputs]

    # ---- 分布式收集结果 ----
    # (vLLM) 不再需要
    # ... (代码已移除) ...

    # ---- 主进程计算准确率并保存 ----
    # (vLLM) 移除 rank == 0 判断
    # if rank == 0:

    # 确保我们有对应的数据
    if len(all_outputs) != len(data):
        print(
            f"CRITICAL ERROR: Mismatch in output count. Expected {len(data)}, Got {len(all_outputs)}"
        )
        # 尽力而为，截断数据以匹配输出
        data = data[: len(all_outputs)]

    totals, corrects, final_output = defaultdict(int), defaultdict(int), []
    for i, (item, raw_output) in enumerate(zip(data, all_outputs)):
        gt = item["answer"]
        match = re.search(
            r"<answer>(.*?)</answer>", raw_output, re.DOTALL
        ) or re.search(r"<answer>(.*)", raw_output, re.DOTALL)
        answer = match.group(1).strip() if match else raw_output.strip()
        is_correct = grade_answer(answer, gt)

        if not is_correct and ":" in answer:
            is_correct = grade_answer(answer[0], gt)

        cf_type = "cf" if item.get("is_cf") else "ncf"
        q_type = item.get("type", "unknown")

        # 统计数量
        for key in ["all", cf_type, q_type, f"{q_type}_{cf_type}"]:
            totals[key] += 1
            if is_correct:
                corrects[key] += 1

        final_output.append(
            {
                "images": item["images"],
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
    print(f"\n--- {dataset_name} 准确率报告 ---")
    accuracies = {}
    for key in sorted(totals.keys()):
        acc = (corrects[key] / totals[key]) * 100 if totals[key] > 0 else 0
        accuracies[f"{key}_accuracy"] = acc
        print(f"{key:<12} : {acc:.2f}% ({corrects[key]}/{totals[key]})")

    # ---- 保存结果 ----
    suffix = "_COT" if args.cot else ""
    prefix = f"-{args.model_prefix}" if args.model_prefix else ""
    filename = f"results_{args.model_name}{prefix}_{dataset_name}{suffix}_vLLM.json"  # 添加 vLLM 后缀
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, filename)

    with open(output_path, "w") as f:
        json.dump({"accuracies": accuracies, "results": final_output}, f, indent=4)

    print(f"\n[INFO] 结果已保存到: {output_path}\n{'-' * 100}")


if __name__ == "__main__":
    main()
