# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Sonomos Traffic Classifier — Model Definition

61→96→48→1 dense network (~10,993 params) for binary classification
of AI provider traffic from TLS/HTTPS metadata features.
"""

import torch
import torch.nn as nn
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) for class-imbalanced binary classification.
    Down-weights well-classified examples, focuses gradient on hard negatives.

    Args:
        alpha: Weight for the positive (minority) class. Default 0.75.
        gamma: Focusing parameter. Higher = more focus on hard examples. Default 2.0.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # probability of correct class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class TrafficClassifier(nn.Module):
    """
    Tiny MLP binary classifier for AI traffic detection.

    Architecture: 61 → 96 → 48 → 1
    Parameters:  ~10,705 (+ 288 for BatchNorm = ~10,993 total, folded at inference)
    Output:      raw logit (apply sigmoid for probability)

    During ONNX export, BatchNorm layers are folded into preceding linear
    layers by tract's optimizer, so runtime cost is pure matmul.
    """

    NUM_FEATURES = 61
    HIDDEN_1 = 96
    HIDDEN_2 = 48

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, self.HIDDEN_1),
            nn.BatchNorm1d(self.HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.HIDDEN_1, self.HIDDEN_2),
            nn.BatchNorm1d(self.HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.HIDDEN_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(AI traffic) in [0, 1]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DistillationLoss(nn.Module):
    """
    Combined loss for XGBoost→MLP knowledge distillation.

    Blends hard label loss (focal) with soft teacher probability loss (MSE).

    Args:
        alpha_hard: Weight for focal loss on true labels.
        alpha_soft: Weight for MSE loss against teacher probabilities.
        focal_alpha: Focal loss class weight.
        focal_gamma: Focal loss focusing parameter.
    """

    def __init__(
        self,
        alpha_hard: float = 0.3,
        alpha_soft: float = 0.7,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha_hard = alpha_hard
        self.alpha_soft = alpha_soft
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.mse = nn.MSELoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hard_loss = self.focal(logits, targets)
        if teacher_probs is None:
            return hard_loss
        student_probs = torch.sigmoid(logits)
        soft_loss = self.mse(student_probs, teacher_probs)
        return self.alpha_hard * hard_loss + self.alpha_soft * soft_loss


def export_onnx(
    model: TrafficClassifier,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """
    Export trained model to ONNX for tract inference.

    Uses opset 17, fixed input shape (1, 40), constant folding enabled.
    Verify with: tract traffic_classifier.onnx -i 1,40,f32 dump
    """
    model.eval()
    dummy_input = torch.randn(1, TrafficClassifier.NUM_FEATURES)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["features"],
        output_names=["logit"],
        do_constant_folding=True,
        dynamic_axes=None,  # fixed shape for tract optimization
    )
    print(f"Exported ONNX model to {output_path}")
    print(f"  Parameters: {model.count_parameters()}")
    print(f"  Input shape: (1, {TrafficClassifier.NUM_FEATURES})")
    print(f"  Output: raw logit (apply sigmoid for probability)")
