import torch
import torch.nn as nn
from typing import Optional
import types

# Copyright 2026 Sonomos, Inc.
# All rights reserved.

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()

class TrafficClassifier(nn.Module):
    NUM_FEATURES = 61
    HIDDEN_1 = 96
    HIDDEN_2 = 48
    def __init__(self, dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, self.HIDDEN_1), nn.BatchNorm1d(self.HIDDEN_1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.HIDDEN_1, self.HIDDEN_2), nn.BatchNorm1d(self.HIDDEN_2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_logit = nn.Linear(self.HIDDEN_2, 1)
        self.head_confidence = nn.Linear(self.HIDDEN_2, 1)
    def forward(self, x):
        h = self.backbone(x)
        logit = self.head_logit(h).squeeze(-1)
        confidence = torch.sigmoid(self.head_confidence(h).squeeze(-1))
        return logit, confidence
    def forward_onnx(self, x):
        h = self.backbone(x)
        logit = self.head_logit(h).squeeze(-1)
        confidence = torch.sigmoid(self.head_confidence(h).squeeze(-1))
        return torch.stack([logit, confidence], dim=-1)
    def predict_proba(self, x):
        with torch.no_grad():
            logit, confidence = self.forward(x)
            return torch.sigmoid(logit), confidence
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

class ConfidenceAwareLoss(nn.Module):
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, confidence_penalty=0.1):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.confidence_penalty = confidence_penalty
    def forward(self, logit, confidence, targets):
        prob = torch.sigmoid(logit)
        prob_adjusted = confidence * prob + (1.0 - confidence) * targets
        eps = 1e-7
        bce = -(targets * torch.log(prob_adjusted + eps) + (1 - targets) * torch.log(1 - prob_adjusted + eps))
        p_t = torch.where(targets == 1, prob_adjusted, 1 - prob_adjusted)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        classification_loss = (focal_weight * bce).mean()
        confidence_loss = -torch.log(confidence + eps).mean()
        return classification_loss + self.confidence_penalty * confidence_loss

class DistillationLoss(nn.Module):
    def __init__(self, alpha_hard=0.3, alpha_soft=0.7, focal_alpha=0.75, focal_gamma=2.0, confidence_penalty=0.1):
        super().__init__()
        self.alpha_hard = alpha_hard
        self.alpha_soft = alpha_soft
        self.conf_loss = ConfidenceAwareLoss(focal_alpha=focal_alpha, focal_gamma=focal_gamma, confidence_penalty=confidence_penalty)
        self.mse = nn.MSELoss()
    def forward(self, logit, confidence, targets, teacher_probs=None):
        hard_loss = self.conf_loss(logit, confidence, targets)
        if teacher_probs is None: return hard_loss
        student_probs = torch.sigmoid(logit)
        soft_loss = self.mse(student_probs, teacher_probs)
        return self.alpha_hard * hard_loss + self.alpha_soft * soft_loss

class _OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model.forward_onnx(x)

def export_onnx(model, output_path, opset_version=17):
    model.eval()
    wrapper = _OnnxWrapper(model)
    wrapper.eval()
    dummy_input = torch.randn(1, TrafficClassifier.NUM_FEATURES)
    torch.onnx.export(wrapper, dummy_input, output_path, opset_version=opset_version,
        input_names=["features"], output_names=["output"],
        do_constant_folding=True, dynamic_axes=None, dynamo=False)
    print(f"Exported ONNX model to {output_path}")
    print(f"  Parameters: {model.count_parameters()}")
    print(f"  Output shape: (1, 2) = [logit, confidence]")
