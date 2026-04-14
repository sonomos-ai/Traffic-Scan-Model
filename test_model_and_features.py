# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""Tests for model definition and feature extraction."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import TrafficClassifier, FocalLoss, DistillationLoss, export_onnx
from features import (
    NUM_FEATURES,
    FEATURE_NAMES,
    FlowRecord,
    FlowStats,
    TLSMetadata,
    JA4Components,
    extract_features,
    sni_ngram_hash,
    flow_to_features,
    tls_to_features,
    ja4_to_features,
    first_n_to_features,
)


# --- Model tests ---


class TestTrafficClassifier:
    def test_parameter_count(self):
        model = TrafficClassifier()
        # 40*64 + 64 + 64*2 + 64*32 + 32 + 32*2 + 32*1 + 1
        # = 2560 + 64 + 128 + 2048 + 32 + 64 + 32 + 1 = 4929 with BN
        # Without BN params counted separately: 2560+64+2048+32+32+1 = 4737
        count = model.count_parameters()
        assert 4700 <= count <= 5000, f"Expected ~4929 params, got {count}"

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

    def test_onnx_export(self, tmp_path):
        model = TrafficClassifier()
        model.eval()
        path = str(tmp_path / "test_model.onnx")
        export_onnx(model, path)

        import onnx

        loaded = onnx.load(path)
        onnx.checker.check_model(loaded)

        # Check opset
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

        # PyTorch output
        with torch.no_grad():
            pt_logit = model(x_torch).item()

        # ONNX output
        onnx_logit = session.run(None, {"features": x_np})[0][0]

        assert abs(pt_logit - onnx_logit) < 1e-4, (
            f"PyTorch ({pt_logit:.6f}) != ONNX ({onnx_logit:.6f})"
        )


class TestFocalLoss:
    def test_zero_loss_for_perfect_prediction(self):
        loss_fn = FocalLoss()
        # logit = +10 (very confident positive) vs target = 1
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
        bce = FocalLoss(gamma=0.0)  # gamma=0 → standard BCE

        logits = torch.tensor([3.0, -3.0])  # easy: correct predictions
        targets = torch.tensor([1.0, 0.0])

        focal_loss = focal(logits, targets)
        bce_loss = bce(logits, targets)
        assert focal_loss < bce_loss, "Focal should down-weight easy examples"


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


# --- Feature tests ---


class TestSNINgramHash:
    def test_output_shape(self):
        vec = sni_ngram_hash("api.openai.com")
        assert vec.shape == (3,)

    def test_normalized(self):
        vec = sni_ngram_hash("api.openai.com")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5 or norm == 0.0

    def test_different_domains_different_hashes(self):
        h1 = sni_ngram_hash("api.openai.com")
        h2 = sni_ngram_hash("www.google.com")
        assert not np.allclose(h1, h2), "Different domains should hash differently"

    def test_deterministic(self):
        h1 = sni_ngram_hash("api.anthropic.com")
        h2 = sni_ngram_hash("api.anthropic.com")
        np.testing.assert_array_equal(h1, h2)

    def test_empty_domain(self):
        vec = sni_ngram_hash("")
        assert vec.shape == (3,)
        assert np.allclose(vec, 0.0)  # too short for any n-grams


class TestFlowFeatures:
    def test_output_shape(self):
        flow = FlowStats(
            packet_sizes=[100, 200, 300],
            inter_arrival_times=[0.01, 0.02],
            duration_seconds=1.0,
            packet_count_upstream=1,
            packet_count_downstream=2,
            total_bytes=600,
        )
        feats = flow_to_features(flow)
        assert feats.shape == (16,)

    def test_empty_flow(self):
        flow = FlowStats()
        feats = flow_to_features(flow)
        assert feats.shape == (16,)
        # Should not crash, all zeros or near-zero

    def test_normalization_range(self):
        flow = FlowStats(
            packet_sizes=[1460] * 100,
            inter_arrival_times=[0.001] * 99,
            duration_seconds=10.0,
            packet_count_upstream=50,
            packet_count_downstream=50,
            total_bytes=146000,
        )
        feats = flow_to_features(flow)
        # Most features should be in [0, 1] range
        assert feats.max() <= 2.0
        assert feats.min() >= -0.1


class TestFirstNFeatures:
    def test_output_shape(self):
        feats = first_n_to_features([100, 200, 300, 400, 500])
        assert feats.shape == (8,)

    def test_padding(self):
        feats = first_n_to_features([100, 200])
        assert feats[2] == 0.0  # padded
        assert feats[7] == 0.0  # padded

    def test_truncation(self):
        feats = first_n_to_features(list(range(20)))
        assert feats.shape == (8,)  # only first 8


class TestTLSFeatures:
    def test_output_shape(self):
        tls = TLSMetadata(version="TLS1.3", cipher_suite_count=15, extension_count=10)
        feats = tls_to_features(tls)
        assert feats.shape == (7,)


class TestJA4Features:
    def test_output_shape(self):
        ja4 = JA4Components(tls_version="TLS1.3", cipher_count=10, extension_count=8)
        feats = ja4_to_features(ja4)
        assert feats.shape == (6,)


class TestExtractFeatures:
    def test_full_extraction(self):
        record = FlowRecord(
            flow_stats=FlowStats(
                packet_sizes=[100, 500, 200, 800],
                inter_arrival_times=[0.01, 0.02, 0.03],
                duration_seconds=2.0,
                packet_count_upstream=2,
                packet_count_downstream=2,
                total_bytes=1600,
                first_n_packet_sizes=[100, 500, 200, 800],
            ),
            tls_metadata=TLSMetadata(
                version="TLS1.3",
                cipher_suite_count=15,
                extension_count=10,
                alpn="h2",
                has_h2_alpn=True,
                cert_chain_length=3,
            ),
            ja4_components=JA4Components(
                tls_version="TLS1.3",
                cipher_count=15,
                extension_count=10,
                alpn="h2",
            ),
            sni_domain="api.openai.com",
        )
        feats = extract_features(record)
        assert feats.shape == (NUM_FEATURES,)
        assert feats.dtype == np.float32

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == NUM_FEATURES
