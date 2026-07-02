# -*- coding: utf-8 -*-
"""
基于 PyTorch 1D-CNN 的心跳信号四分类实验

功能：
1. 构建 1D-CNN 模型
2. 使用加权交叉熵进行训练
3. 在测试集上计算 Accuracy、Macro-F1、分类报告和混淆矩阵
4. 保存模型、训练曲线、混淆矩阵和数据划分编号
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ==============================
# 1. 配置参数
# ==============================

SEED = 42
BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 2e-4
NUM_CLASSES = 4
PATIENCE = 8
VAL_SIZE = 0.1

# 原始反频率权重会让少数类 label=1 权重过大，容易把 label=0 误判成 1。
# 使用幂次平滑后，仍然照顾少数类，但决策边界不会过度偏向少数类。
CLASS_WEIGHT_POWER = 0.5
LABEL_SMOOTHING = 0.03

USE_SAMPLE_NORMALIZATION = True
AUGMENT_TRAIN = True
NOISE_STD = 0.01
SCALE_STD = 0.05

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
CHECKPOINT_DIR = ROOT / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# 2. 固定随机种子
# ==============================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================
# 3. 数据集定义
# ==============================

class HeartbeatDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True,
        augment: bool = False,
    ):
        signals = signals.astype(np.float32, copy=True)

        if normalize:
            means = signals.mean(axis=1, keepdims=True)
            stds = signals.std(axis=1, keepdims=True)
            signals = (signals - means) / (stds + 1e-6)

        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        signal = self.signals[index]

        if self.augment:
            signal = signal.clone()
            signal = signal * (1.0 + torch.randn(1, 1) * SCALE_STD)
            signal = signal + torch.randn_like(signal) * NOISE_STD

        return signal, self.labels[index]


# ==============================
# 4. 1D-CNN 模型
# ==============================

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.main(x) + self.shortcut(x))
        return self.dropout(x)


class HeartbeatCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            ResidualBlock(32, 64, kernel_size=7, stride=2, dropout=0.05),
            ResidualBlock(64, 128, kernel_size=5, stride=2, dropout=0.10),
            ResidualBlock(128, 192, kernel_size=5, stride=2, dropout=0.10),
            ResidualBlock(192, 256, kernel_size=3, stride=2, dropout=0.15),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        x = self.classifier(x)
        return x


# ==============================
# 5. 训练与评价函数
# ==============================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for signals, labels in tqdm(loader, desc="Training", leave=False):
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(signals)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for signals, labels in tqdm(loader, desc="Evaluating", leave=False):
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(signals)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * labels.size(0)

        all_probs.append(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    epoch_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    )

    # 天池原任务使用四类概率与 one-hot 真实标签的绝对误差和
    one_hot_labels = np.eye(NUM_CLASSES)[all_labels]
    abs_sum = np.abs(one_hot_labels - all_probs).sum()
    mean_abs_sum = abs_sum / len(all_labels)

    return {
        "loss": epoch_loss,
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "abs_sum": abs_sum,
        "mean_abs_sum": mean_abs_sum,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }


def build_class_weights(labels: np.ndarray) -> torch.Tensor:
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    raw_weights = len(labels) / (NUM_CLASSES * class_counts)
    smoothed_weights = np.power(raw_weights, CLASS_WEIGHT_POWER)
    smoothed_weights = smoothed_weights / smoothed_weights.mean()
    return torch.tensor(smoothed_weights, dtype=torch.float32).to(DEVICE)


def draw_training_curve(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training Stage: Train vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "train_val_loss_curve.png", dpi=300)
    plt.savefig(FIGURE_DIR / "training_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_f1"], label="Train Macro-F1")
    plt.plot(epochs, history["val_f1"], label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("CNN Training Stage: Train vs Validation Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "train_val_f1_curve.png", dpi=300)
    plt.savefig(FIGURE_DIR / "f1_curve.png", dpi=300)
    plt.close()


def draw_confusion_matrix(cm: np.ndarray) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm / np.maximum(row_sums, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    fig.suptitle("CNN Training Stage: Final Evaluation on Held-out Test Set")

    raw_image = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Test Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_yticks(range(NUM_CLASSES))

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(raw_image, ax=axes[0], fraction=0.046, pad=0.04)

    norm_image = axes[1].imshow(cm_normalized, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Test Row Normalized")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_yticks(range(NUM_CLASSES))

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[1].text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(norm_image, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIGURE_DIR / "test_evaluation_confusion_matrix.png", dpi=300)
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


# ==============================
# 6. 主程序
# ==============================

def main() -> None:
    set_seed(SEED)

    print("========== 实验环境 ==========")
    print(f"设备：{DEVICE}")
    print(f"PyTorch 版本：{torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    x = np.load(PROCESSED_DIR / "signals.npy")
    y = np.load(PROCESSED_DIR / "labels.npy")
    ids = np.load(PROCESSED_DIR / "ids.npy")

    # ==============================
    # 读取已经保存好的数据划分
    # ==============================

    split_file = OUTPUT_DIR / "split_indices.npz"

    if not split_file.exists():
        raise FileNotFoundError(
            "未找到数据划分文件 split_indices.npz。\n"
            "请先运行：python src/split_data.py"
        )

    split_data = np.load(split_file)
    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y[train_idx],
    )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print("\n========== 读取固定数据划分 ==========")
    print(f"划分文件：{split_file}")
    print(f"训练集规模：{len(y_train)}")
    print(f"验证集规模：{len(y_val)}")
    print(f"测试集规模：{len(y_test)}")

    for name, labels in [
        ("TrainSet", y_train),
        ("ValidationSet", y_val),
        ("TestSet", y_test),
    ]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n{name} 类别分布：")
        for label, count in zip(unique, counts):
            print(f"label={label}: {count} ({count / len(labels) * 100:.2f}%)")

    train_dataset = HeartbeatDataset(
        x_train,
        y_train,
        normalize=USE_SAMPLE_NORMALIZATION,
        augment=AUGMENT_TRAIN,
    )
    val_dataset = HeartbeatDataset(
        x_val,
        y_val,
        normalize=USE_SAMPLE_NORMALIZATION,
        augment=False,
    )
    test_dataset = HeartbeatDataset(
        x_test,
        y_test,
        normalize=USE_SAMPLE_NORMALIZATION,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=4,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # num_workers=4,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = build_class_weights(y_train)

    print("\n损失函数类别权重（已平滑）：")
    for label, weight in enumerate(class_weights.cpu().numpy()):
        print(f"label={label}: weight={weight:.4f}")

    model = HeartbeatCNN(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_f1 = -1.0
    epochs_without_improvement = 0
    best_model_path = CHECKPOINT_DIR / "best_cnn_model.pt"

    print("\n========== 开始训练 ==========")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        val_result = evaluate(model, val_loader, criterion)
        scheduler.step(val_result["macro_f1"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_result["loss"])
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_result["macro_f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_result['loss']:.4f} | "
            f"Val Acc: {val_result['accuracy']:.4f} | "
            f"Val F1: {val_result['macro_f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_result["macro_f1"] > best_f1:
            best_f1 = val_result["macro_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  保存当前最优模型：Validation Macro-F1={best_f1:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("触发 Early Stopping，停止训练。")
            break

    draw_training_curve(history)

    # 读取最佳模型并进行最终测试
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    final_result = evaluate(model, test_loader, criterion)

    report = classification_report(
        final_result["labels"],
        final_result["preds"],
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(final_result["labels"], final_result["preds"])
    draw_confusion_matrix(cm)

    print("\n========== 最终测试结果 ==========")
    print(f"Accuracy        : {final_result['accuracy']:.4f}")
    print(f"Macro Precision : {final_result['macro_precision']:.4f}")
    print(f"Macro Recall    : {final_result['macro_recall']:.4f}")
    print(f"Macro F1        : {final_result['macro_f1']:.4f}")
    print(f"ABS-SUM         : {final_result['abs_sum']:.4f}")
    print(f"Mean ABS-SUM    : {final_result['mean_abs_sum']:.6f}")

    print("\n========== 分类报告 ==========")
    print(report)

    print("\n========== 混淆矩阵 ==========")
    print(cm)

    metrics_text = (
        "训练设置\n"
        f"Epochs: {len(history['train_loss'])}/{EPOCHS}\n"
        f"Best Validation Macro F1: {best_f1:.4f}\n"
        f"Class Weight Power: {CLASS_WEIGHT_POWER}\n"
        f"Label Smoothing: {LABEL_SMOOTHING}\n"
        f"Sample Normalization: {USE_SAMPLE_NORMALIZATION}\n"
        f"Train Augmentation: {AUGMENT_TRAIN}\n\n"
        "最终测试结果\n"
        f"Accuracy: {final_result['accuracy']:.4f}\n"
        f"Macro Precision: {final_result['macro_precision']:.4f}\n"
        f"Macro Recall: {final_result['macro_recall']:.4f}\n"
        f"Macro F1: {final_result['macro_f1']:.4f}\n"
        f"ABS-SUM: {final_result['abs_sum']:.4f}\n"
        f"Mean ABS-SUM: {final_result['mean_abs_sum']:.6f}\n\n"
        "分类报告\n"
        f"{report}\n\n"
        "混淆矩阵\n"
        f"{cm}\n"
    )

    with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as file:
        file.write(metrics_text)

    print("\n========== 输出文件 ==========")
    print(f"最佳模型：{best_model_path}")
    print(f"评价结果：{OUTPUT_DIR / 'metrics.txt'}")
    print(f"数据划分：{OUTPUT_DIR / 'split_ids.csv'}")
    print(f"训练曲线：{FIGURE_DIR / 'training_curve.png'}")
    print(f"F1 曲线：{FIGURE_DIR / 'f1_curve.png'}")
    print(f"混淆矩阵：{FIGURE_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
