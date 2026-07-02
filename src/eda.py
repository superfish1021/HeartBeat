# -*- coding: utf-8 -*-
"""
数据分析与可视化：
1. 输出数据基本信息
2. 绘制类别分布柱状图
3. 绘制不同类别的典型心跳波形
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    x = np.load(PROCESSED_DIR / "signals.npy")
    y = np.load(PROCESSED_DIR / "labels.npy")

    print("========== 数据说明 ==========")
    print(f"样本总数：{len(y)}")
    print(f"每条信号长度：{x.shape[1]}")
    print(f"信号最小值：{x.min():.4f}")
    print(f"信号最大值：{x.max():.4f}")

    label_counts = pd.Series(y).value_counts().sort_index()
    label_rates = label_counts / len(y) * 100

    print("\n========== 标签分布 ==========")
    for label in label_counts.index:
        print(
            f"类别 {label}: {label_counts[label]} 条，"
            f"占比 {label_rates[label]:.2f}%"
        )

    # 1. 绘制类别分布柱状图
    plt.figure(figsize=(7, 5))
    plt.bar(label_counts.index.astype(str), label_counts.values)
    plt.xlabel("Label")
    plt.ylabel("Number of samples")
    plt.title("Distribution of Heartbeat Signal Labels")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "label_distribution.png", dpi=300)
    plt.close()

    # 2. 每类绘制一条示例波形
    plt.figure(figsize=(10, 6))
    for label in sorted(np.unique(y)):
        sample_index = np.where(y == label)[0][0]
        plt.plot(x[sample_index], label=f"Label {label}")

    plt.xlabel("Time Point")
    plt.ylabel("Signal Value")
    plt.title("Examples of Heartbeat Signals from Different Classes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "heartbeat_samples.png", dpi=300)
    plt.close()

    print(f"\n图像已保存至：{FIGURE_DIR}")


if __name__ == "__main__":
    main()