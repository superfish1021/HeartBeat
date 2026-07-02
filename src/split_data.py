# -*- coding: utf-8 -*-
"""
数据划分脚本：
1. 读取处理后的 signals.npy、labels.npy 和 ids.npy
2. 按 8:2 分层随机划分训练集和测试集
3. 保存训练集索引、测试集索引
4. 保存 split_ids.csv，便于实验复现和提交 GitHub

说明：
本脚本只需要运行一次。
后续模型训练直接读取已经保存的数据划分结果。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ==============================
# 1. 参数与路径设置
# ==============================

SEED = 42
TEST_SIZE = 0.2

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# 2. 数据划分函数
# ==============================

def main() -> None:
    labels_path = PROCESSED_DIR / "labels.npy"
    ids_path = PROCESSED_DIR / "ids.npy"

    if not labels_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            "未找到处理后的 labels.npy 或 ids.npy，"
            "请先运行：python src/prepare_data.py"
        )

    y = np.load(labels_path)
    ids = np.load(ids_path)

    if len(y) != len(ids):
        raise ValueError("标签数量与样本编号数量不一致。")

    indices = np.arange(len(y))

    # 按 8:2 进行分层随机划分
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # 保存索引文件：训练代码以后直接读取该文件
    split_index_path = OUTPUT_DIR / "split_indices.npz"
    np.savez(
        split_index_path,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    # 使用向量化方法生成 train/test 标记
    # 不再使用低效的列表推导式反复构造 set
    split_labels = np.full(len(y), "test", dtype="<U5")
    split_labels[train_idx] = "train"

    split_df = pd.DataFrame({
        "id": ids,
        "set": split_labels,
        "label": y,
    })

    split_csv_path = OUTPUT_DIR / "split_ids.csv"
    split_df.to_csv(split_csv_path, index=False, encoding="utf-8-sig")

    # 输出整体信息
    print("========== 数据划分完成 ==========")
    print(f"随机种子：{SEED}")
    print(f"总样本数：{len(y)}")
    print(f"训练集样本数：{len(train_idx)}")
    print(f"测试集样本数：{len(test_idx)}")

    print("\n========== 训练集类别分布 ==========")
    train_labels = y[train_idx]
    unique, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        ratio = count / len(train_labels) * 100
        print(f"label={label}: {count} ({ratio:.2f}%)")

    print("\n========== 测试集类别分布 ==========")
    test_labels = y[test_idx]
    unique, counts = np.unique(test_labels, return_counts=True)
    for label, count in zip(unique, counts):
        ratio = count / len(test_labels) * 100
        print(f"label={label}: {count} ({ratio:.2f}%)")

    print("\n========== 保存文件 ==========")
    print(f"索引文件：{split_index_path}")
    print(f"划分记录：{split_csv_path}")


if __name__ == "__main__":
    main()