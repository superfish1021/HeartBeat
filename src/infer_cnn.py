# -*- coding: utf-8 -*-
"""
CNN 心跳信号推理脚本

功能：
1. 读取 split_indices.npz 中保留的 20% 测试集
2. 将测试集整理为 CSV 和 NumPy 文件，保存到 outputs/infer
3. 加载 checkpoints/best_cnn_model.pt 进行推理
4. 输出预测概率、预测类别、错分样本和结果分析报告
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
INFER_DIR = OUTPUT_DIR / "infer"
CHECKPOINT_DIR = ROOT / "checkpoints"
NUM_CLASSES = 4

sys.path.insert(0, str(SRC_DIR))
from train_cnn import HeartbeatCNN, HeartbeatDataset, USE_SAMPLE_NORMALIZATION  # noqa: E402


def parse_args():
    parser = ArgumentParser(description="Run inference on the fixed 20% test split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "best_cnn_model.pt",
        help="Path to trained CNN checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INFER_DIR,
        help="Directory for inference outputs.",
    )
    return parser.parse_args()


def load_fixed_test_split():
    signals_path = PROCESSED_DIR / "signals.npy"
    labels_path = PROCESSED_DIR / "labels.npy"
    ids_path = PROCESSED_DIR / "ids.npy"
    split_path = OUTPUT_DIR / "split_indices.npz"

    for path in [signals_path, labels_path, ids_path, split_path]:
        if not path.exists():
            raise FileNotFoundError(f"缺少必要文件：{path}")

    signals = np.load(signals_path)
    labels = np.load(labels_path)
    ids = np.load(ids_path)
    split_data = np.load(split_path)
    test_idx = split_data["test_idx"]

    return signals[test_idx], labels[test_idx], ids[test_idx], test_idx


def save_test_set(output_dir: Path, signals, labels, ids, test_idx):
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_paths = [
        output_dir / "test_signals.npy",
        output_dir / "test_labels.npy",
        output_dir / "test_ids.npy",
        output_dir / "test_indices.npy",
    ]
    csv_path = output_dir / "test_set.csv"

    if all(path.exists() for path in npy_paths) and csv_path.exists():
        print(f"测试集整理文件已存在，跳过重复写入：{output_dir}")
        return

    np.save(npy_paths[0], signals)
    np.save(npy_paths[1], labels)
    np.save(npy_paths[2], ids)
    np.save(npy_paths[3], test_idx)

    if not csv_path.exists():
        signal_strings = [
            ",".join(f"{value:.8g}" for value in row)
            for row in tqdm(signals, desc="Writing test_set.csv", leave=False)
        ]
        test_df = pd.DataFrame({
            "id": ids,
            "heartbeat_signals": signal_strings,
            "label": labels,
        })
        test_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()

    all_probs = []
    all_preds = []

    for signals, _ in tqdm(loader, desc="Inferencing", leave=False):
        signals = signals.to(device)
        logits = model(signals)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    confidence = probs.max(axis=1)

    return preds, probs, confidence


def save_prediction_files(output_dir: Path, ids, labels, preds, probs, confidence):
    prob_columns = [f"label_{i}" for i in range(NUM_CLASSES)]

    submit_df = pd.DataFrame(probs, columns=prob_columns)
    submit_df.insert(0, "id", ids)
    submit_df.to_csv(
        output_dir / "submit_result.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df = pd.DataFrame({
        "id": ids,
        "true_label": labels,
        "pred_label": preds,
        "confidence": confidence,
    })
    for i, column in enumerate(prob_columns):
        pred_df[column] = probs[:, i]
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]
    pred_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    wrong_df = pred_df[~pred_df["correct"]].copy()
    wrong_df = wrong_df.sort_values(
        by=["true_label", "pred_label", "confidence"],
        ascending=[True, True, False],
    )
    wrong_df.to_csv(
        output_dir / "misclassified.csv",
        index=False,
        encoding="utf-8-sig",
    )

    np.save(output_dir / "pred_probs.npy", probs)
    np.save(output_dir / "pred_labels.npy", preds)

    return pred_df, wrong_df


def plot_confusion_matrix(output_dir: Path, cm: np.ndarray):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm / np.maximum(row_sums, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    fig.suptitle("CNN Inference Stage: Held-out 20% Test Set")

    raw_image = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Inference Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_yticks(range(NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(raw_image, ax=axes[0], fraction=0.046, pad=0.04)

    norm_image = axes[1].imshow(cm_normalized, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Inference Row Normalized")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_yticks(range(NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[1].text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(norm_image, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_dir / "inference_confusion_matrix.png", dpi=300)
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def build_analysis_text(labels, preds, probs, confidence, cm, checkpoint):
    accuracy = accuracy_score(labels, preds)
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    one_hot_labels = np.eye(NUM_CLASSES)[labels]
    abs_sum = np.abs(one_hot_labels - probs).sum()
    mean_abs_sum = abs_sum / len(labels)

    report = classification_report(labels, preds, digits=4, zero_division=0)

    true_unique, true_counts = np.unique(labels, return_counts=True)
    pred_unique, pred_counts = np.unique(preds, return_counts=True)
    true_distribution = {
        int(label): int(count)
        for label, count in zip(true_unique, true_counts)
    }
    pred_distribution = {
        int(label): int(count)
        for label, count in zip(pred_unique, pred_counts)
    }

    error_pairs = []
    for true_label in range(NUM_CLASSES):
        for pred_label in range(NUM_CLASSES):
            if true_label == pred_label:
                continue
            count = int(cm[true_label, pred_label])
            if count > 0:
                error_pairs.append((count, true_label, pred_label))
    error_pairs.sort(reverse=True)

    low_confidence_threshold = 0.6
    low_confidence_count = int((confidence < low_confidence_threshold).sum())

    analysis = {
        "checkpoint": str(checkpoint),
        "num_samples": int(len(labels)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "abs_sum": float(abs_sum),
        "mean_abs_sum": float(mean_abs_sum),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "low_confidence_threshold": low_confidence_threshold,
        "low_confidence_count": low_confidence_count,
        "true_distribution": true_distribution,
        "pred_distribution": pred_distribution,
        "confusion_matrix": cm.astype(int).tolist(),
        "top_error_pairs": [
            {"count": count, "true_label": true_label, "pred_label": pred_label}
            for count, true_label, pred_label in error_pairs[:10]
        ],
    }

    lines = [
        "CNN 推理结果分析",
        "",
        f"模型权重: {checkpoint}",
        f"测试样本数: {len(labels)}",
        f"Accuracy: {accuracy:.4f}",
        f"Macro Precision: {macro_precision:.4f}",
        f"Macro Recall: {macro_recall:.4f}",
        f"Macro F1: {macro_f1:.4f}",
        f"Weighted F1: {weighted_f1:.4f}",
        f"ABS-SUM: {abs_sum:.4f}",
        f"Mean ABS-SUM: {mean_abs_sum:.6f}",
        f"平均置信度: {confidence.mean():.4f}",
        f"置信度低于 {low_confidence_threshold:.1f} 的样本数: {low_confidence_count}",
        "",
        "真实类别分布",
    ]
    for label in range(NUM_CLASSES):
        lines.append(f"label={label}: {true_distribution.get(label, 0)}")

    lines.extend(["", "预测类别分布"])
    for label in range(NUM_CLASSES):
        lines.append(f"label={label}: {pred_distribution.get(label, 0)}")

    lines.extend(["", "分类报告", report, "", "混淆矩阵", str(cm)])

    if error_pairs:
        lines.extend(["", "主要错分方向"])
        for count, true_label, pred_label in error_pairs[:10]:
            lines.append(f"true={true_label} -> pred={pred_label}: {count}")
    else:
        lines.extend(["", "主要错分方向", "无错分样本"])

    return "\n".join(lines) + "\n", analysis


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    signals, labels, ids, test_idx = load_fixed_test_split()
    labels = labels.astype(np.int64)

    save_test_set(output_dir, signals, labels, ids, test_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HeartbeatDataset(
        signals,
        labels,
        normalize=USE_SAMPLE_NORMALIZATION,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = HeartbeatCNN(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    preds, probs, confidence = run_inference(model, loader, device)
    pred_df, wrong_df = save_prediction_files(
        output_dir,
        ids,
        labels,
        preds,
        probs,
        confidence,
    )

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    plot_confusion_matrix(output_dir, cm)

    analysis_text, analysis = build_analysis_text(
        labels,
        preds,
        probs,
        confidence,
        cm,
        args.checkpoint,
    )
    (output_dir / "analysis.txt").write_text(analysis_text, encoding="utf-8")
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("========== 推理完成 ==========")
    print(f"测试集样本数: {len(labels)}")
    print(f"错分样本数: {len(wrong_df)}")
    print(f"预测明细: {output_dir / 'predictions.csv'}")
    print(f"提交概率: {output_dir / 'submit_result.csv'}")
    print(f"结果分析: {output_dir / 'analysis.txt'}")
    print(f"混淆矩阵: {output_dir / 'inference_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
