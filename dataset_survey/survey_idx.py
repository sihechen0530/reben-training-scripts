import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import typer


DEFAULT_INDEX_PATH = Path("BENv2_index.json")


def guess_label_field(entry: dict) -> str:
    """从单个 entry 里猜 label 字段名."""
    cand_keys = ["labels", "label_names", "lc_labels", "benv2_labels"]
    for k in cand_keys:
        if k in entry:
            return k
    raise KeyError(f"找不到标签字段，entry keys = {list(entry.keys())}")


def _normalize_labels(raw_labels: Iterable[str] | str) -> list[str]:
    if isinstance(raw_labels, str):
        return [raw_labels]
    if isinstance(raw_labels, (tuple, set)):
        return list(raw_labels)
    return list(raw_labels)


def summarize_index(index: dict, label_field: str, top_k: int) -> dict:
    label_counter = Counter()
    labels_per_sample: list[int] = []

    for patch_id, info in index.items():
        if label_field not in info:
            raise KeyError(f"Entry {patch_id} 缺少标签字段 {label_field!r}")
        labels = _normalize_labels(info[label_field])
        label_counter.update(labels)
        labels_per_sample.append(len(labels))

    labels_per_sample_np = np.array(labels_per_sample)
    labels_hist = Counter(labels_per_sample)

    summary = {
        "num_samples": len(index),
        "num_unique_labels": len(label_counter),
        "labels_per_sample": {
            "min": int(labels_per_sample_np.min()),
            "max": int(labels_per_sample_np.max()),
            "mean": float(labels_per_sample_np.mean()),
            "median": float(np.median(labels_per_sample_np)),
        },
        "labels_per_sample_hist": dict(sorted(labels_hist.items())),
        "top_labels": label_counter.most_common(top_k),
        "rare_labels": sorted(label_counter.items(), key=lambda x: x[1])[:top_k],
    }
    return summary


def print_summary(summary: dict, sample_key: str, entry_keys: list[str], label_field: str) -> None:
    print("索引样本数量:", summary["num_samples"])
    print("标签种类数:", summary["num_unique_labels"])
    print("示例样本 ID:", sample_key)
    print("示例 entry keys:", entry_keys)
    print("使用的标签字段:", label_field)

    print("\n=== 每个样本标签数统计 ===")
    stats = summary["labels_per_sample"]
    print("min:", stats["min"], "max:", stats["max"], "mean:", f"{stats['mean']:.3f}", "median:", f"{stats['median']:.3f}")
    print("标签数直方图(标签个数 -> 样本数):")
    for label_cnt, sample_cnt in summary["labels_per_sample_hist"].items():
        print(f"  {label_cnt:2d} -> {sample_cnt:7d}")

    print("\n=== 最常见的标签 ===")
    for lab, cnt in summary["top_labels"]:
        print(f"{lab:40s}  {cnt:7d}")

    print("\n=== 最稀有的标签 ===")
    for lab, cnt in summary["rare_labels"]:
        print(f"{lab:40s}  {cnt:7d}")


def main(
        index_path: Path = typer.Option(
            DEFAULT_INDEX_PATH,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="BENv2_index.json 的完整路径"
        ),
        top_k: int = typer.Option(10, min=1, help="显示最常见/最稀有标签的数量"),
        export_json: Optional[Path] = typer.Option(
            None,
            help="如果提供，则把汇总统计保存为 JSON 文件"
        ),
) -> None:
    with index_path.open("r") as f:
        index = json.load(f)

    first_key = next(iter(index))
    first_entry = index[first_key]
    label_field = guess_label_field(first_entry)
    summary = summarize_index(index, label_field, top_k)
    print_summary(summary, first_key, list(first_entry.keys()), label_field)

    if export_json is not None:
        export_json.parent.mkdir(parents=True, exist_ok=True)
        with export_json.open("w") as f:
            json.dump({
                "index_path": str(index_path.resolve()),
                "label_field": label_field,
                **summary,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n已将汇总结果保存到 {export_json}")


if __name__ == "__main__":
    typer.run(main)