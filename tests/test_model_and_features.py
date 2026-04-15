# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""Tests for two-head model definition (requires torch)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import TrafficClassifier, FocalLoss, ConfidenceAwareLoss, DistillationLoss, export_onnx
from features import NUM_FEATURES


class TestTrafficClassifier:
    def test_parameter_count(self):
        model = TrafficClassifier()
        count = model.count_parameters()
        # Backbone: same as before ~10.9K
        # + confidence head: 48*1 + 1 = 49 extra params
        assert 10000 <= count <= 12500, f"Expected ~11K params, got {count}"

    def test_forward_returns_two_heads(self):
        model = TrafficClassifier()
        x = torch.randn(16, NUM_FEATURES)
        logit, confidence = model(x)
        assert logit.shape == (16,), f"Expected logit (16,), got {logit.shape}"
        assert confidence.shape == (16,), f"Expected confidence (16,), got {confidence.shape}"

    def test_confidence_in_range(self):
        model = TrafficClassifier()
        x = torch.randn(100, NUM_FEATURES)
        _, confidence = model(x)
        assert confidence.min() >= 0.0, "Confidence should be >= 0"
        assert confidence.max() <= 1.0, "Confidence should be <= 1"

    def test_predict_proba_returns_two_values(self):
        model = TrafficClassifier()
        x = torch.randn(10, NUM_FEATURES)
        probs, conf = model.predict_proba(x)
        assert probs.min() >= 0.0 and probs.max() <= 1.0
        assert conf.min() >= 0.0 and conf.max() <= 1.0

    def test_single_sample(self):
        model = TrafficClassifier()
        x = torch.randn(1, NUM_FEATURES)
        logit, conf = model(x)
        assert logit.shape == (1,)
        assert conf.shape == (1,)

    def test_deterministic_eval(self):
        model = TrafficClassifier()
        model.eval()
        x = torch.randn(1, NUM_FEATURES)
        l1, c1 = model(x)
        l2, c2 = model(x)
        assert l1.item() == l2.item()
        assert c1.item() == c2.item()

    def test_num_features_matches(self):
        assert TrafficClassifier.NUM_FEATURES == NUM_FEATURES == 61

    def test_forward_onnx_shape(self):
        model = TrafficClassifier()
        model.eval()
        x = torch.randn(1, NUM_FEATURES)
        out = model.forward_onnx(x)
        assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"

    def test_forward_onnx_values(self):
        model = TrafficClassifier()
        model.eval()
        x = torch.randn(1, NUM_FEATURES)
        logit, conf = model(x)
        onnx_out = model.forward_onnx(x)
        assert abs(onnx_out[0, 0].item() - logit.item()) < 1e-6
        assert abs(onnx_out[0, 1].item() - conf.item()) < 1e-6

    def test_onnx_export(self, tmp_path):
        model = TrafficClassifier()
        model.eval()
        path = str(tmp_path / "test_model.onnx")
        export_onnx(model, path)

        import onnx
        loaded = onnx.load(path)
        onnx.checker.check_model(loaded)
        assert loaded.opset_import[0].version == 17

    def test_onnx_output_shape(self, tmp_path):
        model = TrafficClassifier()
        model.eval()
        path = str(tmp_path / "test_model.onnx")
        export_onnx(model, path)

        import onnxruntime as ort
        session = ort.InferenceSession(path)
        x_np = np.random.randn(1, NUM_FEATURES).astype(np.float32)
        out = session.run(None, {"features": x_np})[0]
        assert out.shape == (1, 2), f"Expected ONNX output (1, 2), got {out.shape}"

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
            pt_out = model.forward_onnx(x_torch).numpy()

        onnx_out = session.run(None, {"features": x_np})[0]

        assert abs(pt_out[0, 0] - onnx_out[0, 0]) < 1e-4, (
            f"Logit mismatch: PyTorch={pt_out[0,0]:.6f}, ONNX={onnx_out[0,0]:.6f}"
        )
        assert abs(pt_out[0, 1] - onnx_out[0, 1]) < 1e-4, (
            f"Confidence mismatch: PyTorch={pt_out[0,1]:.6f}, ONNX={onnx_out[0,1]:.6f}"
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
        assert focal(logits, targets) < bce(logits, targets)


class TestConfidenceAwareLoss:
    def test_high_confidence_correct_prediction(self):
        loss_fn = ConfidenceAwareLoss(confidence_penalty=0.1)
        logit = torch.tensor([5.0])    # confident positive
        conf = torch.tensor([0.95])    # high confidence
        target = torch.tensor([1.0])   # correct
        loss = loss_fn(logit, conf, target)
        assert loss.item() > 0  # should be small but positive

    def test_low_confidence_reduces_classification_loss(self):
        loss_fn = ConfidenceAwareLoss(confidence_penalty=0.01)  # low penalty
        logit = torch.tensor([-5.0])   # wrong: predicts negative
        target = torch.tensor([1.0])   # actual: positive

        # High confidence on wrong prediction → high loss
        high_conf = torch.tensor([0.95])
        loss_high = loss_fn(logit, high_conf, target)

        # Low confidence → model peeks at label → lower classification loss
        low_conf = torch.tensor([0.1])
        loss_low = loss_fn(logit, low_conf, target)

        # With low penalty, the classification savings should dominate
        assert loss_low < loss_high, (
            f"Low confidence ({loss_low:.4f}) should have lower loss "
            f"than high confidence on wrong prediction ({loss_high:.4f})"
        )

    def test_budget_penalty_prevents_zero_confidence(self):
        loss_fn = ConfidenceAwareLoss(confidence_penalty=1.0)  # strong penalty
        logit = torch.tensor([0.0])
        conf = torch.tensor([0.01])   # near-zero confidence
        target = torch.tensor([1.0])
        loss = loss_fn(logit, conf, target)
        # -log(0.01) ≈ 4.6, times penalty 1.0 → significant
        assert loss.item() > 3.0

    def test_gradient_flows_to_confidence_head(self):
        model = TrafficClassifier()
        loss_fn = ConfidenceAwareLoss()
        x = torch.randn(8, NUM_FEATURES)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        logit, conf = model(x)
        loss = loss_fn(logit, conf, targets)
        loss.backward()

        # Confidence head should receive gradients
        assert model.head_confidence.weight.grad is not None
        assert model.head_confidence.weight.grad.abs().sum() > 0


class TestDistillationLoss:
    def test_without_teacher(self):
        loss_fn = DistillationLoss()
        logit = torch.tensor([1.0, -1.0])
        conf = torch.tensor([0.8, 0.8])
        targets = torch.tensor([1.0, 0.0])
        loss = loss_fn(logit, conf, targets, teacher_probs=None)
        assert loss.item() > 0

    def test_with_teacher(self):
        loss_fn = DistillationLoss()
        logit = torch.tensor([1.0, -1.0])
        conf = torch.tensor([0.8, 0.8])
        targets = torch.tensor([1.0, 0.0])
        teacher = torch.tensor([0.9, 0.1])
        loss = loss_fn(logit, conf, targets, teacher_probs=teacher)
        assert loss.item() > 0
