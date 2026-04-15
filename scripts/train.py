# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Training script for the Sonomos Traffic Classifier.

Supports:
  - Standard focal loss training
  - XGBoost teacher distillation (--teacher flag)
  - Cosine annealing LR schedule
  - Stratified k-fold cross-validation
  - Temporal split validation (--temporal-split)
  - ONNX export with tract-compatible settings

Usage:
    # Standard training
    python scripts/train.py --data data/train.csv --output models/traffic_classifier.onnx

    # With XGBoost distillation
    python scripts/train.py --data data/train.csv --teacher models/xgb_teacher.json --output models/traffic_classifier.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import TrafficClassifier, FocalLoss, ConfidenceAwareLoss, DistillationLoss, export_onnx
from features import NUM_FEATURES, FEATURE_NAMES


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and labels from CSV."""
    import csv

    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    X = np.array([[float(v) for v in row[:NUM_FEATURES]] for row in rows], dtype=np.float32)
    y = np.array([float(row[NUM_FEATURES]) for row in rows], dtype=np.float32)
    return X, y


def load_teacher_probs(path: str, X: np.ndarray) -> np.ndarray:
    """Load XGBoost teacher and generate soft probability targets."""
    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    teacher = xgb.Booster()
    teacher.load_model(path)
    dmat = xgb.DMatrix(X)
    probs = teacher.predict(dmat)
    print(f"Loaded teacher from {path}, generated {len(probs)} soft targets")
    return probs.astype(np.float32)


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency sample weights for WeightedRandomSampler."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    weight_pos = len(y) / (2 * max(n_pos, 1))
    weight_neg = len(y) / (2 * max(n_neg, 1))
    weights = np.where(y == 1, weight_pos, weight_neg)
    return weights


def train_epoch(
    model: TrafficClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    teacher_probs: dict[int, torch.Tensor] | None = None,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (X_batch, y_batch, idx_batch) in enumerate(loader):
        optimizer.zero_grad()
        logits, confidence = model(X_batch)

        if teacher_probs is not None and isinstance(criterion, DistillationLoss):
            t_probs = teacher_probs[idx_batch]
            loss = criterion(logits, confidence, y_batch, t_probs)
        else:
            loss = criterion(logits, confidence, y_batch)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: TrafficClassifier,
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    """Evaluate model, return metrics dict including confidence stats."""
    model.eval()
    logits, confidence = model(X)
    probs = torch.sigmoid(logits).cpu().numpy()
    conf = confidence.cpu().numpy()
    y_np = y.cpu().numpy()

    preds = (probs >= 0.5).astype(float)

    metrics = {
        "auc_pr": average_precision_score(y_np, probs) if y_np.sum() > 0 else 0.0,
        "auc_roc": roc_auc_score(y_np, probs) if len(np.unique(y_np)) > 1 else 0.0,
        "f1": f1_score(y_np, preds, zero_division=0),
        "conf_mean": float(conf.mean()),
        "conf_std": float(conf.std()),
    }

    # Confidence calibration: mean confidence on correct vs incorrect predictions
    correct_mask = (preds == y_np)
    if correct_mask.any():
        metrics["conf_correct"] = float(conf[correct_mask].mean())
    if (~correct_mask).any():
        metrics["conf_incorrect"] = float(conf[~correct_mask].mean())

    # Precision at 90% recall
    if y_np.sum() > 0:
        precision, recall, _ = precision_recall_curve(y_np, probs)
        mask = recall >= 0.9
        metrics["precision_at_90_recall"] = float(precision[mask].max()) if mask.any() else 0.0
    else:
        metrics["precision_at_90_recall"] = 0.0

    return metrics


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    teacher_probs_train: np.ndarray | None = None,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
) -> tuple[TrafficClassifier, dict]:
    """
    Full training loop with early stopping.

    Returns trained model and best validation metrics.
    """
    # Convert to tensors
    indices_train = np.arange(len(X_train))
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    idx_t = torch.tensor(indices_train, dtype=torch.long)

    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)

    # Weighted sampler for class imbalance
    sample_weights = compute_sample_weights(y_train)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True,
    )

    dataset = TensorDataset(X_t, y_t, idx_t)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Model and optimizer
    model = TrafficClassifier(dropout=0.1)
    print(f"Model parameters: {model.count_parameters()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Loss function
    if teacher_probs_train is not None:
        criterion = DistillationLoss(
            alpha_hard=0.3, alpha_soft=0.7,
            focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            confidence_penalty=0.1,
        )
        t_probs_tensor = torch.tensor(teacher_probs_train, dtype=torch.float32)
    else:
        criterion = ConfidenceAwareLoss(
            focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            confidence_penalty=0.1,
        )
        t_probs_tensor = None

    # Training loop with early stopping
    best_auc_pr = 0.0
    best_state = None
    best_metrics = {}
    no_improve = 0

    for epoch in range(epochs):
        loss = train_epoch(model, loader, criterion, optimizer, t_probs_tensor)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate(model, X_v, y_v)
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch+1:3d} | loss={loss:.4f} | "
                f"AUC-PR={metrics['auc_pr']:.4f} | F1={metrics['f1']:.4f} | "
                f"P@R90={metrics['precision_at_90_recall']:.4f} | "
                f"conf={metrics['conf_mean']:.3f}±{metrics['conf_std']:.3f} | lr={lr_now:.2e}"
            )

            if metrics["auc_pr"] > best_auc_pr:
                best_auc_pr = metrics["auc_pr"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_metrics = metrics.copy()
                no_improve = 0
            else:
                no_improve += 10

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train traffic classifier")
    parser.add_argument("--data", type=str, required=True, help="Training CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--teacher", type=str, default=None, help="XGBoost teacher model path")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--folds", type=int, default=5, help="K-fold CV folds (0 = single 80/20 split)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-split", action="store_true",
                        help="Use temporal split (first 80%% train, last 20%% val)")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    X, y = load_csv(args.data)
    print(f"  Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"  Positive (AI): {int(y.sum())} ({y.mean()*100:.1f}%)")
    print(f"  Negative (normal): {int(len(y) - y.sum())} ({(1-y.mean())*100:.1f}%)")

    # Load teacher if specified
    teacher_probs = None
    if args.teacher:
        teacher_probs = load_teacher_probs(args.teacher, X)

    # Validation strategy
    if args.temporal_split:
        print("\nUsing temporal split (80/20)...")
        split_idx = int(len(X) * 0.8)
        train_idx = np.arange(split_idx)
        val_idx = np.arange(split_idx, len(X))
        splits = [(train_idx, val_idx)]
    elif args.folds > 0:
        print(f"\nUsing {args.folds}-fold stratified CV...")
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(X, y))
    else:
        print("\nUsing single 80/20 split...")
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(X))
        split_idx = int(len(X) * 0.8)
        splits = [(perm[:split_idx], perm[split_idx:])]

    # Train (use last fold for final model)
    all_metrics = []
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold+1}/{len(splits)} ---")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        t_probs = teacher_probs[train_idx] if teacher_probs is not None else None

        model, metrics = train(
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            teacher_probs_train=t_probs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
        )
        all_metrics.append(metrics)
        best_model = model

    # Summary
    print("\n=== Results ===")
    for key in ["auc_pr", "auc_roc", "f1", "precision_at_90_recall"]:
        values = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Export ONNX
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(best_model, str(output_path))

    # Save metrics
    metrics_path = output_path.with_suffix(".metrics.json")
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg_metrics["n_folds"] = len(splits)
    avg_metrics["n_samples"] = len(y)
    avg_metrics["n_positive"] = int(y.sum())
    avg_metrics["parameters"] = best_model.count_parameters()
    with open(metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
