import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


INDEX_PATH = Path("BENv2_index.json")


def guess_label_field(entry: dict) -> str:
    cand_keys = ["labels", "label_names", "lc_labels", "benv2_labels"]
    for k in cand_keys:
        if k in entry:
            return k
    raise KeyError(f"找不到标签字段，entry keys = {list(entry.keys())}")


def guess_split_field(entry: dict) -> str:
    cand_keys = ["split", "subset", "partition", "set"]
    for k in cand_keys:
        if k in entry:
            return k
    raise KeyError(f"找不到 split 字段，entry keys = {list(entry.keys())}")


def main() -> None:
    with INDEX_PATH.open("r") as f:
        index = json.load(f)

    print("样本数量:", len(index))

    first_key = next(iter(index))
    first_entry = index[first_key]
    print("\n=== 示例 entry keys ===", list(first_entry.keys()))

    label_field = guess_label_field(first_entry)
    split_field = guess_split_field(first_entry)

    print(f"使用的标签字段: {label_field!r}")
    print(f"使用的 split 字段: {split_field!r}")

    # 每个 split 一份 Counter
    split_label_counter: dict[str, Counter] = defaultdict(Counter)
    split_sample_counts: Counter = Counter()
    split_labels_per_sample: dict[str, list[int]] = defaultdict(list)

    for _, info in index.items():
        split = info.get(split_field, "unknown")

        labels = info[label_field]
        if isinstance(labels, str):
            labels = [labels]
        elif isinstance(labels, (tuple, set)):
            labels = list(labels)

        split_sample_counts[split] += 1
        split_labels_per_sample[split].append(len(labels))
        split_label_counter[split].update(labels)

    print("\n=== 各 split 样本数 ===")
    for split, cnt in split_sample_counts.items():
        print(f"{split:10s}: {cnt:6d}")

    for split, counter in split_label_counter.items():
        labels_per_sample = np.array(split_labels_per_sample[split])

        print(f"\n====== Split: {split} ======")
        print("样本数:", split_sample_counts[split])
        print("标签总数:", len(counter))
        print("每样本标签数: min", int(labels_per_sample.min()),
              "max", int(labels_per_sample.max()),
              "mean", float(labels_per_sample.mean()))

        print("\n  最常见的前 5 个标签:")
        for lab, cnt in counter.most_common(5):
            print(f"    {lab:40s}  {cnt:6d}")

        print("\n  最稀有的 5 个标签:")
        for lab, cnt in sorted(counter.items(), key=lambda x: x[1])[:5]:
            print(f"    {lab:40s}  {cnt:6d}")


if __name__ == "__main__":
    main()