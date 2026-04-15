# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Sonomos Traffic Classifier — Two-Head Model Definition

61→96→48→{logit, confidence} dense network (~11K params) for binary
classification of AI provider traffic from TLS/HTTPS metadata features.

Two outputs:
  - logit:      raw classification score (sigmoid → P(AI traffic))
  - confidence: learned confidence score in [0, 1] indicating how much
                the model trusts its own prediction. Low confidence
                signals "I don't know" and the pipeline should fall back
                to a more conservative action.

The confidence head is trained using the DeVries & Taylor (2018) trick:
the prediction is interpolated between the model's output and the true
label, weighted by confidence. This gives the model two ways to minimize
loss: (a) predict correctly with high confidence, or (b) admit uncertainty
with low confidence. A budget penalty prevents trivial c=0 solutions.
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
    Two-head MLP binary classifier for AI traffic detection.

    Architecture: 61 → 96 → 48 → {logit(1), confidence(1)}
    Parameters:  ~11K (+ BatchNorm, folded at inference by tract)

    Outputs (forward):
        logit:      raw classification score, shape (batch,)
        confidence: learned confidence in [0, 1], shape (batch,)

    During ONNX export, both outputs are concatenated into a single
    (1, 2) tensor: [logit, confidence].
    """

    NUM_FEATURES = 61
    HIDDEN_1 = 96
    HIDDEN_2 = 48

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, self.HIDDEN_1),
            nn.BatchNorm1d(self.HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.HIDDEN_1, self.HIDDEN_2),
            nn.BatchNorm1d(self.HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Classification head: raw logit
        self.head_logit = nn.Linear(self.HIDDEN_2, 1)
        # Confidence head: learned confidence score
        self.head_confidence = nn.Linear(self.HIDDEN_2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logit = self.head_logit(h).squeeze(-1)
        confidence = torch.sigmoid(self.head_confidence(h).squeeze(-1))
        return logit, confidence

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        """
        ONNX-compatible forward: returns (1, 2) tensor [logit, confidence].
        Used only during export — tract reads both values from a single output.
        """
        logit, confidence = self.forward(x)
        return torch.stack([logit, confidence], dim=-1)

    def predict_proba(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (P(AI traffic), confidence) both in [0, 1]."""
        with torch.no_grad():
            logit, confidence = self.forward(x)
            return torch.sigmoid(logit), confidence

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ConfidenceAwareLoss(nn.Module):
    """
    Confidence-aware loss (DeVries & Taylor, 2018).

    The model outputs a prediction p and confidence c. The effective
    prediction is interpolated: p' = c * p + (1 - c) * y, where y is
    the true label. This gives the model two ways to minimize loss:
      (a) predict correctly with high confidence
      (b) admit uncertainty with low confidence (peek at the label)

    A budget penalty -log(c) prevents the trivial c=0 solution.

    Args:
        focal_alpha: Focal loss class weight for the classification term.
        focal_gamma: Focal loss focusing parameter.
        confidence_penalty: Weight for the -log(c) budget term. Higher
            values force the model to be more confident. Start with 0.1.
    """

    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        confidence_penalty: float = 0.1,
    ):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.confidence_penalty = confidence_penalty

    def forward(
        self,
        logit: torch.Tensor,
        confidence: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Classification probability
        prob = torch.sigmoid(logit)

        # Interpolated prediction: if confident, use own prediction;
        # if not, peek at the true label
        prob_adjusted = confidence * prob + (1.0 - confidence) * targets

        # Focal-weighted BCE on the adjusted prediction
        # We compute BCE manually since prob_adjusted is already a probability
        eps = 1e-7
        bce = -(targets * torch.log(prob_adjusted + eps)
                + (1 - targets) * torch.log(1 - prob_adjusted + eps))

        p_t = torch.where(targets == 1, prob_adjusted, 1 - prob_adjusted)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        classification_loss = (focal_weight * bce).mean()

        # Budget penalty: prevent trivial c=0 by penalizing low confidence
        confidence_loss = -torch.log(confidence + eps).mean()

        return classification_loss + self.confidence_penalty * confidence_loss


class DistillationLoss(nn.Module):
    """
    Combined loss for XGBoost→MLP knowledge distillation with confidence.

    Blends hard label loss (confidence-aware focal) with soft teacher
    probability loss (MSE on the classification head only).

    Args:
        alpha_hard: Weight for confidence-aware focal loss on true labels.
        alpha_soft: Weight for MSE loss against teacher probabilities.
        focal_alpha: Focal loss class weight.
        focal_gamma: Focal loss focusing parameter.
        confidence_penalty: Weight for confidence budget penalty.
    """

    def __init__(
        self,
        alpha_hard: float = 0.3,
        alpha_soft: float = 0.7,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        confidence_penalty: float = 0.1,
    ):
        super().__init__()
        self.alpha_hard = alpha_hard
        self.alpha_soft = alpha_soft
        self.conf_loss = ConfidenceAwareLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            confidence_penalty=confidence_penalty,
        )
        self.mse = nn.MSELoss()

    def forward(
        self,
        logit: torch.Tensor,
        confidence: torch.Tensor,
        targets: torch.Tensor,
        teacher_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hard_loss = self.conf_loss(logit, confidence, targets)
        if teacher_probs is None:
            return hard_loss
        # Mask out sentinel values (-1) from augmented samples without teacher targets
        valid_mask = teacher_probs >= 0
        if valid_mask.any():
            student_probs = torch.sigmoid(logit[valid_mask])
            soft_loss = self.mse(student_probs, teacher_probs[valid_mask])
        else:
            soft_loss = torch.tensor(0.0)
        return self.alpha_hard * hard_loss + self.alpha_soft * soft_loss


class _OnnxWrapper(nn.Module):
    """Thin wrapper that calls forward_onnx so ONNX export gets a single (1,2) tensor."""

    def __init__(self, model: TrafficClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_onnx(x)


def export_onnx(
    model: TrafficClassifier,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """
    Export trained model to ONNX for tract inference.

    Output shape is (1, 2): [logit, confidence].
    Apply sigmoid to logit for P(AI). Confidence is already in [0, 1].
    """
    model.eval()
    dummy_input = torch.randn(1, TrafficClassifier.NUM_FEATURES)
    wrapper = _OnnxWrapper(model)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["features"],
        output_names=["output"],
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )

    print(f"Exported ONNX model to {output_path}")
    print(f"  Parameters: {model.count_parameters()}")
    print(f"  Input shape: (1, {TrafficClassifier.NUM_FEATURES})")
    print(f"  Output shape: (1, 2) = [logit, confidence]")
