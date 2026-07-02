# -*- coding: utf-8 -*-
"""
数据预处理：
1. 读取 train.csv
2. 将 heartbeat_signals 字符串转为长度 205 的数值数组
3. 将标签转为整数
4. 保存为 numpy 文件，便于后续快速训练
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_signals(signal_series: pd.Series) -> np.ndarray:
    """将逗号分隔的字符串信号转换为 float32 数组。"""
    signals = [
        np.asarray(signal.split(","), dtype=np.float32)
        for signal in signal_series
    ]
    return np.stack(signals)


def main() -> None:
    print(f"正在读取数据：{CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    required_columns = {"id", "heartbeat_signals", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"数据必须包含列：{required_columns}")

    print(f"原始数据规模：{df.shape}")

    x = parse_signals(df["heartbeat_signals"])
    y = df["label"].astype(np.int64).to_numpy()
    ids = df["id"].to_numpy()

    if x.shape[1] != 205:
        raise ValueError(f"信号长度异常，当前长度为：{x.shape[1]}")

    print(f"信号矩阵形状：{x.shape}")
    print(f"标签形状：{y.shape}")
    print(f"信号取值范围：[{x.min():.4f}, {x.max():.4f}]")

    unique, counts = np.unique(y, return_counts=True)
    print("\n类别分布：")
    for label, count in zip(unique, counts):
        print(f"label={label}: {count} ({count / len(y) * 100:.2f}%)")

    np.save(OUTPUT_DIR / "signals.npy", x)
    np.save(OUTPUT_DIR / "labels.npy", y)
    np.save(OUTPUT_DIR / "ids.npy", ids)

    print(f"\n处理完成，结果已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()