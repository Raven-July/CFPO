import json

# === 参数设置 ===
json_path_1 = "/home/u2021213615/share/yzy/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_result/results_Qwen2.5-VL-3B-cmve-a1-MARS-fold1_MARS_Bench_COT_.json"  # 第一个 JSON 文件路径
json_path_2 = "/home/u2021213615/share/yzy/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_result/results_Qwen2.5-VL-3B-MARS-fold1_MARS_Bench_COT_.json"  # 第二个 JSON 文件路径
output_path = "./diff_results.json"  # 输出文件路径

# === 加载 JSON 数据 ===
with open(json_path_1, "r", encoding="utf-8") as f1:
    data1 = json.load(f1)["results"]

with open(json_path_2, "r", encoding="utf-8") as f2:
    data2 = json.load(f2)["results"]

# === 构建 (image, question) -> item 的索引 ===
dict2 = {(item["image"], item["question"]): item for item in data2}

# === 对比逻辑 ===
diffs = []
for item1 in data1:
    key = (item1["image"], item1["question"])
    item2 = dict2.get(key)
    if not item2:
        continue

    if item1.get("correct") == 1 and item2.get("correct") == 0:
        diffs.append(
            {
                "image": item1.get("image", ""),
                "question": item1.get("question", ""),
                "ground_truth": item1.get("ground_truth", ""),
                "output1": item1.get("model_output", ""),
                "output2": item2.get("model_output", ""),
            }
        )

# === 输出结果 ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(diffs, f, indent=4, ensure_ascii=False)

print(
    f"✅ 对比完成，共发现 {len(diffs)} 条在文件1正确但文件2错误的样本。结果已保存至 {output_path}"
)
