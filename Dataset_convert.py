import json
import os
import random
from collections import defaultdict

# 输入输出路径
# input_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_data/MARS_Bench.json"
# output_train_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/MARS_Bench_train.json"
# output_test_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/MARS_Bench_test.json"
# # 为原始格式的测试集定义输出路径
# output_test_original_format_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/MARS_Bench_test_original.json"
# output_distribution_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/MARS_Bench_train_distribution.json"

input_path = "/home/u2021213615/share/yzy/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_data/C-VQA-Synthetic.json"
output_train_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/C-VQA-Synthetic_train.json"
output_test_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/C-VQA-Synthetic_test.json"
# 为原始格式的测试集定义输出路径
output_test_original_format_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/C-VQA-Synthetic_test_original.json"
output_distribution_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/C-VQA-Synthetic_train_distribution.json"


random_seed = 42  # 固定随机种子
TRAIN_SET_SIZE = 600  # 设置训练集样本总数
# marsbench 5110 3600
# cvqa-real 6288 1800
# cvqa-Synthetic 6000 600


def convert_data(input_path):
    """
    读取 json 文件，转换为目标格式，并同时返回原始数据。
    """
    with open(input_path, "r") as f:
        original_data = json.load(f)

    converted = []
    for item in original_data:
        converted.append(
            {
                "images": [item["image"]],  # 保持和原代码一致，放在list里
                "problem": "<image>" + item["problem"],
                "answer": item["solution"],
                "id": str(item["id"]),
                "type": item.get("type", "unknown"),  # 确保type存在，避免None作为key
                "is_cf": item.get("is_cf", False),  # 将None视为False
            }
        )
    # 返回原始数据和转换后的数据
    return original_data, converted


def main():
    # 1. 读取并转换数据，同时获取原始数据
    original_data, converted_data = convert_data(input_path)

    # 2. 设置随机种子
    random.seed(random_seed)

    # 3. 按类型和is_cf对数据进行分组
    data_by_type = defaultdict(lambda: {"cf": [], "non_cf": []})
    for item in converted_data:
        if item["is_cf"]:
            data_by_type[item["type"]]["cf"].append(item)
        else:
            data_by_type[item["type"]]["non_cf"].append(item)

    # 对每个子列表进行随机排序
    for type_key in data_by_type:
        random.shuffle(data_by_type[type_key]["cf"])
        random.shuffle(data_by_type[type_key]["non_cf"])

    # 4. 计算每个type在训练集中的目标数量，以实现均衡
    all_types = sorted(data_by_type.keys())  # 排序以保证确定性
    num_types = len(all_types)
    base_count_per_type = TRAIN_SET_SIZE // num_types
    remainder = TRAIN_SET_SIZE % num_types

    type_targets = {}
    for i, type_key in enumerate(all_types):
        type_targets[type_key] = base_count_per_type + (1 if i < remainder else 0)

    # 5. 迭代构建训练集
    train_data = []
    train_ids = set()

    # 记录每个类别已经使用了多少数据
    used_indices = {type_key: {"cf": 0, "non_cf": 0} for type_key in all_types}

    # 实时统计训练集中各类别数量
    current_train_counts = {
        type_key: {"total": 0, "cf": 0, "non_cf": 0} for type_key in all_types
    }

    while len(train_data) < TRAIN_SET_SIZE:
        best_category_to_add = None
        best_score = -1

        # 遍历所有可能的选择，找到最优的一个
        for type_key in all_types:
            # a. 计算该类型的不平衡度（分数越高，代表越需要补充）
            type_fullness = (
                current_train_counts[type_key]["total"] / type_targets[type_key]
                if type_targets[type_key] > 0
                else 1
            )
            type_score = 1.0 - type_fullness

            # b. 检查is_cf类别
            current_total = current_train_counts[type_key]["total"]
            current_cf = current_train_counts[type_key]["cf"]

            if used_indices[type_key]["cf"] < len(data_by_type[type_key]["cf"]):
                error_if_add_cf = abs(((current_cf + 1) / (current_total + 1)) - 0.8)
                cf_score = type_score + (0.8 - error_if_add_cf)
                if cf_score > best_score:
                    best_score = cf_score
                    best_category_to_add = (type_key, "cf")

            # c. 检查non_cf类别
            if used_indices[type_key]["non_cf"] < len(data_by_type[type_key]["non_cf"]):
                error_if_add_non_cf = abs((current_cf / (current_total + 1)) - 0.8)
                non_cf_score = type_score + (0.8 - error_if_add_non_cf)
                if non_cf_score > best_score:
                    best_score = non_cf_score
                    best_category_to_add = (type_key, "non_cf")

        if best_category_to_add:
            sel_type, sel_cat = best_category_to_add
            item_index = used_indices[sel_type][sel_cat]
            selected_item = data_by_type[sel_type][sel_cat][item_index]
            train_data.append(selected_item)
            train_ids.add(selected_item["id"])

            used_indices[sel_type][sel_cat] += 1
            current_train_counts[sel_type]["total"] += 1
            current_train_counts[sel_type][sel_cat] += 1
        else:
            break

    # 6. 构建两种格式的测试集
    # a. 转换后格式的测试集
    test_data = [item for item in converted_data if item["id"] not in train_ids]
    # b. 新增：原始文件格式的测试集
    test_data_original_format = [
        item for item in original_data if item.get("id") not in train_ids
    ]

    # 7. 最后将训练集随机打乱，使其内部顺序是无序的
    random.shuffle(train_data)

    # 8. 保存所有文件
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)

    # 保存训练集
    with open(output_train_path, "w") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    # 保存转换后格式的测试集
    with open(output_test_path, "w") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    # 保存原始格式的测试集
    with open(output_test_original_format_path, "w") as f:
        json.dump(test_data_original_format, f, indent=4, ensure_ascii=False)

    print(f"转换完成！训练集共 {len(train_data)} 条数据，已保存到 {output_train_path}")
    print(
        f"转换完成！转换后格式测试集共 {len(test_data)} 条数据，已保存到 {output_test_path}"
    )
    print(
        f"转换完成！原始格式测试集共 {len(test_data_original_format)} 条数据，已保存到 {output_test_original_format_path}"
    )

    # (可选) 打印训练集中的详细分布以供验证
    print("\n训练集详细分布情况:")
    distribution_data = {}
    for type_key in all_types:
        counts = current_train_counts[type_key]
        total = counts["total"]
        cf_count = counts["cf"]
        non_cf_count = counts["non_cf"]
        ratio = cf_count / non_cf_count if non_cf_count > 0 else float("inf")
        print(
            f"  - 类型 '{type_key}': {total} 条 (目标: {type_targets[type_key]}) | is_cf: {cf_count}, not_is_cf: {non_cf_count} | 比例: {ratio:.2f}:1"
        )
        distribution_data[type_key] = {
            "total": total,
            "target": type_targets[type_key],
            "cf_count": cf_count,
            "non_cf_count": non_cf_count,
            "ratio": f"{ratio:.2f}:1",
        }

    # 保存训练集分布数据到 JSON 文件
    os.makedirs(os.path.dirname(output_distribution_path), exist_ok=True)
    with open(output_distribution_path, "w") as f:
        json.dump(distribution_data, f, indent=4, ensure_ascii=False)
    print(f"训练集分布情况已保存到 {output_distribution_path}")


if __name__ == "__main__":
    main()
