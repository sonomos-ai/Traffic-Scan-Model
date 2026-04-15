# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Train an XGBoost teacher model for knowledge distillation.

The teacher is unconstrained (500 trees, depth 6) to maximize accuracy.
Its soft probability outputs are then used as training targets for the
tiny MLP student, combining the teacher's capacity with the student's
fast inference via tract.

Usage:
    python scripts/train_xgboost_teacher.py --data data/train.csv --output models/xgb_teacher.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import NUM_FEATURES

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and labels from CSV."""
    import csv

    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = list(reader)

    X = np.array([[float(v) for v in row[:NUM_FEATURES]] for row in rows], dtype=np.float32)
    y = np.array([float(row[NUM_FEATURES]) for row in rows], dtype=np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost teacher")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    X, y = load_csv(args.data)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"  Samples: {len(y)}, Pos: {n_pos}, Neg: {n_neg}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": args.seed,
        "tree_method": "hist",
        "verbosity": 0,
    }

    print(f"\nTraining XGBoost ({args.folds}-fold CV)...")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
        dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

        model = xgb.train(
            {k: v for k, v in params.items() if k not in ("n_estimators", "random_state")},
            dtrain,
            num_boost_round=args.n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        probs = model.predict(dval)
        preds = (probs >= 0.5).astype(float)

        metrics = {
            "auc_pr": average_precision_score(y[val_idx], probs),
            "auc_roc": roc_auc_score(y[val_idx], probs),
            "f1": f1_score(y[val_idx], preds, zero_division=0),
        }
        all_metrics.append(metrics)
        print(f"  Fold {fold+1}: AUC-PR={metrics['auc_pr']:.4f} F1={metrics['f1']:.4f}")

    # Summary
    print("\n=== Teacher Results ===")
    for key in ["auc_pr", "auc_roc", "f1"]:
        values = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Train final model on full data
    print("\nTraining final teacher on full dataset...")
    dfull = xgb.DMatrix(X, label=y)
    final_model = xgb.train(
        {k: v for k, v in params.items() if k not in ("n_estimators", "random_state")},
        dfull,
        num_boost_round=args.n_estimators,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(output_path))
    print(f"Saved teacher to {output_path}")

    # Feature importance
    importance = final_model.get_score(importance_type="gain")
    if importance:
        print("\nTop 10 features by gain:")
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, gain in sorted_imp:
            print(f"  {feat}: {gain:.2f}")


if __name__ == "__main__":
    main()
