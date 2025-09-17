import json
import os
import random

# 输入输出路径
input_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Counterfactual-R1/Counterfactual-Eval/eval_data/MARS_Bench.json"
output_train_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/MARS_Bench_train_fold1.json"
output_test_path = "/eaas/default/groups/xitucheng213/home/u2021213615/share/yzy/Counterfact-Projects/Datasets/MARS_Bench_test_fold1.json"
random_seed = 42  # 固定随机种子


def convert_data(input_path):
    """
    读取 json 文件并转换为目标格式
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    converted = []
    for item in data:
        converted.append(
            {
                "images": [item["image"]],  # 保持和原代码一致，放在list里
                "problem": "<image>" + item["problem"],
                "answer": item["solution"],
                "id": item["id"],
                # "incorrect": item.get("incorrect", None),
                # "type": item.get("type", None),
                # "is_cf": item.get("is_cf", None),
            }
        )

    return converted


def main():
    converted_data = convert_data(input_path)

    # 设置随机种子
    random.seed(random_seed)
    random.shuffle(converted_data)

    # 划分训练集和测试集
    train_data = converted_data[:880]
    test_data = converted_data[880:]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)

    # 保存训练集
    with open(output_train_path, "w") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    # 保存测试集
    with open(output_test_path, "w") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"转换完成！训练集共 {len(train_data)} 条数据，已保存到 {output_train_path}")
    print(f"转换完成！测试集共 {len(test_data)} 条数据，已保存到 {output_test_path}")


if __name__ == "__main__":
    main()
