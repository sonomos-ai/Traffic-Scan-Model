# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""Tests for model definition (requires torch)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import TrafficClassifier, FocalLoss, DistillationLoss, export_onnx
from features import NUM_FEATURES


class TestTrafficClassifier:
    def test_parameter_count(self):
        model = TrafficClassifier()
        count = model.count_parameters()
        # 61*96 + 96 + 96*2 + 96*48 + 48 + 48*2 + 48*1 + 1
        # = 5856 + 96 + 192 + 4608 + 48 + 96 + 48 + 1 = 10945
        # Plus BN: +192 +96 = 11233
        assert 10000 <= count <= 12000, f"Expected ~11K params, got {count}"

    def test_forward_shape(self):
        model = TrafficClassifier()
        x = torch.randn(16, NUM_FEATURES)
        out = model(x)
        assert out.shape == (16,), f"Expected (16,), got {out.shape}"

    def test_predict_proba_range(self):
        model = TrafficClassifier()
        x = torch.randn(100, NUM_FEATURES)
        probs = model.predict_proba(x)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_single_sample(self):
        model = TrafficClassifier()
        x = torch.randn(1, NUM_FEATURES)
        out = model(x)
        assert out.shape == (1,)

    def test_deterministic_eval(self):
        model = TrafficClassifier()
        model.eval()
        x = torch.randn(1, NUM_FEATURES)
        out1 = model(x).item()
        out2 = model(x).item()
        assert out1 == out2, "Eval mode should be deterministic"

    def test_num_features_matches(self):
        assert TrafficClassifier.NUM_FEATURES == NUM_FEATURES == 61

    def test_onnx_export(self, tmp_path):
        model = TrafficClassifier()
        model.eval()
        path = str(tmp_path / "test_model.onnx")
        export_onnx(model, path)

        import onnx

        loaded = onnx.load(path)
        onnx.checker.check_model(loaded)
        assert loaded.opset_import[0].version == 17

    def test_onnx_inference_matches_pytorch(self, tmp_path):
        model = TrafficClassifier()
        model.eval()
        path = str(tmp_path / "test_model.onnx")
        export_onnx(model, path)

        import onnxruntime as ort

        session = ort.InferenceSession(path)
        x_np = np.random.randn(1, NUM_FEATURES).astype(np.float32)
        x_torch = torch.tensor(x_np)

        with torch.no_grad():
            pt_logit = model(x_torch).item()

        onnx_logit = session.run(None, {"features": x_np})[0][0]

        assert abs(pt_logit - onnx_logit) < 1e-4, (
            f"PyTorch ({pt_logit:.6f}) != ONNX ({onnx_logit:.6f})"
        )


class TestFocalLoss:
    def test_zero_loss_for_perfect_prediction(self):
        loss_fn = FocalLoss()
        logits = torch.tensor([10.0])
        targets = torch.tensor([1.0])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.01

    def test_high_loss_for_wrong_prediction(self):
        loss_fn = FocalLoss()
        logits = torch.tensor([10.0])
        targets = torch.tensor([0.0])
        loss = loss_fn(logits, targets)
        assert loss.item() > 1.0

    def test_focal_lower_than_bce_for_easy_examples(self):
        focal = FocalLoss(gamma=2.0)
        bce = FocalLoss(gamma=0.0)

        logits = torch.tensor([3.0, -3.0])
        targets = torch.tensor([1.0, 0.0])

        focal_loss = focal(logits, targets)
        bce_loss = bce(logits, targets)
        assert focal_loss < bce_loss


class TestDistillationLoss:
    def test_without_teacher(self):
        loss_fn = DistillationLoss()
        logits = torch.tensor([1.0, -1.0])
        targets = torch.tensor([1.0, 0.0])
        loss = loss_fn(logits, targets, teacher_probs=None)
        assert loss.item() > 0

    def test_with_teacher(self):
        loss_fn = DistillationLoss()
        logits = torch.tensor([1.0, -1.0])
        targets = torch.tensor([1.0, 0.0])
        teacher = torch.tensor([0.9, 0.1])
        loss = loss_fn(logits, targets, teacher_probs=teacher)
        assert loss.item() > 0
