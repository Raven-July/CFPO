import os
import re
import json
import math
import random
import argparse
import warnings
from collections import defaultdict
from typing import Optional, Dict, Any
from PIL import Image
from concurrent.futures import ThreadPoolExecutor  # 引入线程池

from tqdm import tqdm
from transformers import AutoProcessor
from mathruler.grader import grade_answer,extract_boxed_content
from vllm import LLM, SamplingParams

# ----------------- 全局配置 -----------------
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
MIN_PIXELS = 200704  # 最小像素数
MAX_PIXELS = 1003520  # 最大像素数

# vLLM 需要这个环境变量
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
        "--config_path",
        type=str,
        required=True,
        help="包含数据集配置的 JSON 文件的路径",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="保存结果的目录")
    parser.add_argument(
        "--cot", action="store_true", help="是否启用 CoT（链式思维）推理"
    )
    parser.add_argument(
        "--papo", action="store_true", help="是否使用papo测试"
    )
    # 显著减小默认值，因为大数据集的图像加载会爆 RAM
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="BATCH_SIZE for vLLM inference (default: 1024)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM 使用的 GPU 数量 (默认: all available)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="用于并行数据加载的线程数",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Total number of attempts per sample (acc@n). If n > 1, temperature will be set to 1.0 automatically."
    )
    return parser.parse_args()


# ===================== 图像预处理 =====================
def process_image(
    image: Image.Image, min_pixels: Optional[int], max_pixels: Optional[int]
) -> Image.Image:
    """根据最小/最大像素约束调整图像大小，并转换为 RGB"""
    # 缩小到最大像素
    if max_pixels and image.width * image.height > max_pixels:
        factor = math.sqrt(max_pixels / (image.width * image.height))
        # PIL.Image.resize 接受 (width, height)
        image = image.resize((int(image.width * factor), int(image.height * factor)))

    # 放大到最小像素
    if min_pixels and image.width * image.height < min_pixels:
        factor = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))

    if image.mode != "RGB":
        if image.mode == "P":
            # 尝试通过 RGBA 转换为 RGB 以处理透明度/调色板
            image = image.convert("RGBA").convert("RGB")
        else:
            image = image.convert("RGB")
    return image


def load_and_preprocess_single_item(
    item: Dict[str, Any],
    image_root: str,
    processor: AutoProcessor,
    prompt_template: str,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """并行加载、处理单个数据项（图像和提示）"""
    problem_text = item["problem"]
    image_files = item["images"]

    # 1. 为这个 item 加载 PIL 图像
    item_images_pil = []
    for img_name in image_files:
        image_path = os.path.join(image_root, img_name)
        try:
            # 在单独的线程中，Image.open 是安全的
            # 立即转换为 RGB 避免后续问题，并进行缩放/调整
            img = Image.open(image_path)
            img = process_image(img, MIN_PIXELS, MAX_PIXELS)
            item_images_pil.append(img)
        except Exception as e:
            # print(f"Warning: Failed to load image {image_path}: {e}") # 线程中避免过多打印
            # 添加一个占位符灰色图像以避免崩溃
            item_images_pil.append(Image.new("RGB", (448, 448), "grey"))

    # 2. 构建 'messages' 列表（使用虚拟图像路径）
    content_list = []
    parts = problem_text.split("<image>")
    # 确保图像数量匹配
    if len(image_files) != problem_text.count("<image>"):
        # print(f"Warning: Mismatch in <image> tags count for item id {item.get('id', 'N/A')}. Expected {len(image_files)}, Found {problem_text.count('<image>')}")
        pass  # 避免线程过多打印

    img_index = 0
    for i, part in enumerate(parts):
        if i != 0 and img_index < len(item_images_pil):
            # 占位符 token，类型必须是 "image"
            content_list.append({"type": "image"})
            img_index += 1

        if part:
            content_list.append({"type": "text", "text": part})
    if args.papo:
        content_list.append({"type": "text", "text": "You first think through the reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \\boxed{}."})
        messages_for_item = [
            {"role": "user", "content": content_list},
        ]
    else:
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
        # print(f"Error applying template for item {item.get('id', 'N/A')}: {e}")
        return None  # 无法处理，跳过

    # 4. 存储 vLLM 需要的统一输入字典
    # 这里的 `original_item` 不再是字典，而是原始数据列表的索引
    return {
        "prompt": final_prompt_text,
        "multi_modal_data": {"image": item_images_pil},
        "original_item": item,  # 包含原始信息的字典，用于结果保存
    }


# ===================== 核心评估逻辑 =====================
def run_evaluation(
    args: argparse.Namespace,
    llm: LLM,
    processor: AutoProcessor,
    dataset_config: Dict[str, str],
):
    """对单个数据集执行评估流程，计算标准的 avg@n"""
    ds_path = dataset_config["json_path"]
    image_root = dataset_config["image_root"]
    dataset_name = dataset_config["dataset_name"]

    prompt_template = (
        "The user asks a question, and then you solve it based on the images provided. "
        "You FIRST think about the reasoning process step by step as an internal monologue "
        "and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer MUST BE put in \\boxed{}."
        if args.cot else 
        "The user asks a question, and then you solve it based on the images provided. "
        "Please directly give out the final answer, which MUST BE put in \\boxed{}"
    )

    print(f"\n[INFO] Starting evaluation for dataset: **{dataset_name}**")
    
    with open(ds_path, "r") as f:
        data = json.load(f)

    # -------------------------------------------------------
    # 核心修改 1: SamplingParams 的 n 设置为 args.n
    # -------------------------------------------------------
    sampling_params = SamplingParams(
        n=args.n,  # 一次性生成 n 个结果
        temperature=1.0 if args.n > 1 else 0.0,
        max_tokens=2048,
        stop=["<|eot_id|>", "</s>"],
    )

    all_final_results = []
    BATCH_SIZE = args.batch_size
    num_samples = len(data)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="Processing Batches"):
            batch_data = data[i : i + BATCH_SIZE]
            
            # 并行预处理图像和 Prompt
            futures = [
                executor.submit(load_and_preprocess_single_item, item, image_root, processor, prompt_template, args)
                for item in batch_data
            ]
            
            vllm_inputs = []
            valid_items = []
            for idx, future in enumerate(futures):
                res = future.result()
                if res:
                    vllm_inputs.append(res)
                    valid_items.append(batch_data[idx])

            if not vllm_inputs:
                continue

            # -------------------------------------------------------
            # 核心修改 2: vLLM 推理（每个 prompt 会得到 n 个 output）
            # -------------------------------------------------------
            outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params)
            
            for item, out in zip(valid_items, outputs):
                sample_res = {
                    "images": item["images"],
                    "question": item["problem"],
                    "ground_truth": item["answer"],
                    "is_cf": item.get("is_cf", False),
                    "type": item.get("type", "unknown"),
                    "rollouts": []
                }
                
                correct_count = 0
                # 遍历这 n 个采样结果
                for r_idx, completion in enumerate(out.outputs):
                    raw_output = completion.text
                    gt = item["answer"]
                    
                    # 提取与评分
                    answer = extract_boxed_content(raw_output)
                    if not args.papo and answer == "None":
                        answer = raw_output.strip()
                    
                    is_correct = grade_answer(answer, gt)
                    # 额外补正逻辑
                    if not is_correct and not args.papo and isinstance(answer, str) and ":" in answer:
                        is_correct = grade_answer(answer.split(":")[0].strip(), gt)
                    
                    if is_correct:
                        correct_count += 1
                    
                    # 动态保存每一个 rollout 的结果
                    sample_res["rollouts"].append({
                        f"model_output_{r_idx+1}": raw_output,
                        f"extracted_answer_{r_idx+1}": answer,
                        f"score_{r_idx+1}": int(is_correct)
                    })
                
                # 计算该样本的平均分 (e.g., 对了 2 次，则是 2/8 = 0.25)
                sample_res["avg_score"] = correct_count / args.n
                all_final_results.append(sample_res)

    # ---- 统计与保存 ----
    totals, sum_avg_scores = defaultdict(int), defaultdict(float)

    for res in all_final_results:
        cf_type = "cf" if res["is_cf"] else "ncf"
        q_type = res["type"]
        avg_score = res["avg_score"]

        for key in ["all", cf_type, q_type, f"{q_type}_{cf_type}"]:
            totals[key] += 1
            sum_avg_scores[key] += avg_score

    # ---- 计算并打印准确率 ----
    print(f"\n--- {dataset_name} Avg@{args.n} 准确率报告 ---")
    accuracies = {}
    for key in sorted(totals.keys()):
        # 计算该类别所有样本 avg_score 的平均值
        final_acc = (sum_avg_scores[key] / totals[key]) * 100 if totals[key] > 0 else 0
        accuracies[f"{key}_avg@{args.n}_acc"] = final_acc
        print(f"{key:<12} : {final_acc:.2f}% (Total samples: {totals[key]})")

    # ---- 保存详细 JSON ----
    suffix = f"_avg@{args.n}"
    if args.cot: suffix += "_COT"
    prefix = f"-{args.model_prefix}" if args.model_prefix else ""
    filename = f"results_{args.model_name}{prefix}_{dataset_name}{suffix}.json"
    output_path = os.path.join(args.output_dir, filename)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"accuracies": accuracies, "details": all_final_results}, f, indent=4)

    print(f"\n[INFO] 结果已保存到: {output_path}")


# ===================== 主流程 =====================
def main():
    args = parse_args()

    # 确定 GPU 数量
    # world_size = args.tensor_parallel_size if args.tensor_parallel_size else torch.cuda.device_count()

    # ---- 加载数据集配置 ----
    with open(args.config_path, "r") as f:
        datasets_config = json.load(f)

    # ---- 模型和 Processor 加载 (只加载一次) ----
    print(
        f"[INFO] Loading model: {args.model_name} from {args.model_path} with TP={args.tensor_parallel_size}"
    )
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # ---- 循环评估所有数据集 ----
    for ds_config in datasets_config:
        try:
            run_evaluation(args, llm, processor, ds_config)
        except Exception as e:
            print(
                f"[CRITICAL] Error evaluating {ds_config.get('dataset_name', 'Unknown')}: {e}"
            )


if __name__ == "__main__":
    main()
