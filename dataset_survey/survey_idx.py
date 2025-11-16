import json
from collections import Counter
from pathlib import Path

import numpy as np


# ====== 配置区域：如果 BENv2_index.json 不在当前目录，可以改这里 ======
INDEX_PATH = Path("BENv2_index.json")


def guess_label_field(entry: dict) -> str:
    """从单个 entry 里猜 label 字段名."""
    cand_keys = ["labels", "label_names", "lc_labels", "benv2_labels"]
    for k in cand_keys:
        if k in entry:
            return k
    # 如果没有常见字段，打印一下帮你 debug
    raise KeyError(f"找不到标签字段，entry keys = {list(entry.keys())}")


def main() -> None:
    # 1. 读 index
    with INDEX_PATH.open("r") as f:
        index = json.load(f)

    print("索引类型:", type(index))
    print("样本数量:", len(index))

    # 2. 看一个样本长啥样
    first_key = next(iter(index))
    first_entry = index[first_key]
    print("\n=== 示例样本 ===")
    print("示例 key:", first_key)
    print("示例 entry keys:", list(first_entry.keys()))

    # 3. 猜标签字段名
    label_field = guess_label_field(first_entry)
    print(f"\n使用的标签字段名: {label_field!r}")

    # 4. 统计标签分布 & 每个样本标签数
    label_counter = Counter()
    labels_per_sample = []

    for patch_id, info in index.items():
        labels = info[label_field]

        # labels 可能是单个字符串，也可能是 list
        if isinstance(labels, str):
            labels = [labels]
        elif isinstance(labels, (tuple, set)):
            labels = list(labels)

        labels_per_sample.append(len(labels))
        label_counter.update(labels)

    # 5. 打印标签级别统计
    print("\n=== 标签总数 ===")
    print(len(label_counter))

    print("\n=== 最常见的前 10 个标签 ===")
    for lab, cnt in label_counter.most_common(10):
        print(f"{lab:40s}  {cnt:6d}")

    print("\n=== 最稀有的 10 个标签 ===")
    for lab, cnt in sorted(label_counter.items(), key=lambda x: x[1])[:10]:
        print(f"{lab:40s}  {cnt:6d}")

    # 6. 每个样本的标签数统计
    labels_per_sample = np.array(labels_per_sample)
    print("\n=== 每个样本标签数统计 ===")
    print("min:", int(labels_per_sample.min()))
    print("max:", int(labels_per_sample.max()))
    print("mean:", float(labels_per_sample.mean()))


if __name__ == "__main__":
    main()