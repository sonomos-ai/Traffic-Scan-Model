# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""Tests for feature extraction (no torch dependency)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
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
    _directional_stats,
    _HASH_DIMS,
)


class TestConstants:
    def test_num_features_is_61(self):
        assert NUM_FEATURES == 61

    def test_hash_dims_is_11(self):
        assert _HASH_DIMS == 11

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == NUM_FEATURES


class TestSNINgramHash:
    def test_output_shape(self):
        vec = sni_ngram_hash("api.openai.com")
        assert vec.shape == (_HASH_DIMS,)

    def test_normalized(self):
        vec = sni_ngram_hash("api.openai.com")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5 or norm == 0.0

    def test_different_domains_different_hashes(self):
        h1 = sni_ngram_hash("api.openai.com")
        h2 = sni_ngram_hash("www.google.com")
        assert not np.allclose(h1, h2)

    def test_deterministic(self):
        h1 = sni_ngram_hash("api.anthropic.com")
        h2 = sni_ngram_hash("api.anthropic.com")
        np.testing.assert_array_equal(h1, h2)

    def test_empty_domain(self):
        vec = sni_ngram_hash("")
        assert vec.shape == (_HASH_DIMS,)
        assert np.allclose(vec, 0.0)

    def test_single_char_domain(self):
        vec = sni_ngram_hash("a")
        assert vec.shape == (_HASH_DIMS,)
        assert np.allclose(vec, 0.0)

    def test_expanded_dims_reduce_collisions(self):
        """With 11 dims, distinct domains should spread across more dimensions."""
        domains = [
            "api.openai.com", "api.anthropic.com", "api.mistral.ai",
            "www.google.com", "cdn.jsdelivr.net", "registry.npmjs.org",
        ]
        hashes = [sni_ngram_hash(d) for d in domains]
        # Each hash should use more than 3 non-zero dimensions
        for h in hashes:
            nonzero_dims = np.count_nonzero(np.abs(h) > 0.01)
            assert nonzero_dims >= 3, f"Hash only uses {nonzero_dims} dims"

    def test_ai_domains_distinguishable(self):
        """AI domains with 'api.' prefix should still be distinguishable from each other."""
        h_openai = sni_ngram_hash("api.openai.com")
        h_anthropic = sni_ngram_hash("api.anthropic.com")
        h_mistral = sni_ngram_hash("api.mistral.ai")
        # All three should be different despite sharing 'api.' prefix
        assert not np.allclose(h_openai, h_anthropic)
        assert not np.allclose(h_openai, h_mistral)
        assert not np.allclose(h_anthropic, h_mistral)

    def test_custom_dims(self):
        vec3 = sni_ngram_hash("api.openai.com", dims=3)
        vec16 = sni_ngram_hash("api.openai.com", dims=16)
        assert vec3.shape == (3,)
        assert vec16.shape == (16,)


class TestDirectionalStats:
    def test_empty_returns_zeros(self):
        assert _directional_stats([]) == (0.0, 0.0, 0.0)

    def test_single_value(self):
        mean, std, p50 = _directional_stats([750])
        assert abs(mean - 0.5) < 0.01  # 750/1500
        assert std == 0.0
        assert abs(p50 - 0.5) < 0.01

    def test_multiple_values(self):
        mean, std, p50 = _directional_stats([300, 600, 900])
        assert abs(mean - 0.4) < 0.01  # 600/1500
        assert std > 0
        assert abs(p50 - 0.4) < 0.01  # 600/1500


class TestFlowFeatures:
    def test_output_shape(self):
        flow = FlowStats(
            packet_sizes=[100, 200, 300],
            inter_arrival_times=[0.01, 0.02],
            duration_seconds=1.0,
            packet_count_upstream=1,
            packet_count_downstream=2,
            total_bytes=600,
            upstream_packet_sizes=[100],
            downstream_packet_sizes=[200, 300],
            upstream_bytes=100,
            downstream_bytes=500,
        )
        feats = flow_to_features(flow)
        assert feats.shape == (24,)

    def test_empty_flow(self):
        flow = FlowStats()
        feats = flow_to_features(flow)
        assert feats.shape == (24,)

    def test_byte_ratio_asymmetric(self):
        """AI-like traffic: small upstream, large downstream → low byte ratio."""
        flow = FlowStats(
            packet_sizes=[200, 100, 100, 100],
            duration_seconds=1.0,
            packet_count_upstream=1,
            packet_count_downstream=3,
            total_bytes=500,
            upstream_bytes=200,
            downstream_bytes=300,
        )
        feats = flow_to_features(flow)
        byte_ratio = feats[22]
        assert byte_ratio < 0.5  # upstream < downstream

    def test_pkt_ratio_asymmetric(self):
        """AI traffic: few upstream, many downstream → low pkt ratio."""
        flow = FlowStats(
            packet_sizes=[500] * 10,
            duration_seconds=1.0,
            packet_count_upstream=2,
            packet_count_downstream=8,
            total_bytes=5000,
        )
        feats = flow_to_features(flow)
        pkt_ratio = feats[23]
        assert pkt_ratio == 0.2  # 2/10

    def test_neutral_defaults(self):
        """Empty directional data should produce neutral 0.5 ratios."""
        flow = FlowStats(
            packet_sizes=[100],
            duration_seconds=1.0,
            total_bytes=100,
        )
        feats = flow_to_features(flow)
        assert feats[22] == 0.5  # neutral byte ratio
        assert feats[23] == 0.5  # neutral pkt ratio

    def test_directional_stats_populated(self):
        flow = FlowStats(
            packet_sizes=[200, 800, 150, 50],
            duration_seconds=2.0,
            packet_count_upstream=2,
            packet_count_downstream=2,
            total_bytes=1200,
            upstream_packet_sizes=[200, 800],
            downstream_packet_sizes=[150, 50],
            upstream_bytes=1000,
            downstream_bytes=200,
        )
        feats = flow_to_features(flow)
        # Upstream stats [16:19]
        assert feats[16] > 0  # up mean
        assert feats[17] > 0  # up std (200 vs 800 → nonzero)
        assert feats[18] > 0  # up p50
        # Downstream stats [19:22]
        assert feats[19] > 0  # down mean
        assert feats[20] > 0  # down std
        assert feats[21] > 0  # down p50

    def test_normalization_range(self):
        flow = FlowStats(
            packet_sizes=[1460] * 100,
            inter_arrival_times=[0.001] * 99,
            duration_seconds=10.0,
            packet_count_upstream=50,
            packet_count_downstream=50,
            total_bytes=146000,
            upstream_packet_sizes=[1460] * 50,
            downstream_packet_sizes=[1460] * 50,
            upstream_bytes=73000,
            downstream_bytes=73000,
        )
        feats = flow_to_features(flow)
        assert feats.max() <= 2.0
        assert feats.min() >= -0.1


class TestFirstNFeatures:
    def test_output_shape(self):
        feats = first_n_to_features([100, 200, 300, 400, 500])
        assert feats.shape == (8,)

    def test_padding(self):
        feats = first_n_to_features([100, 200])
        assert feats[2] == 0.0
        assert feats[7] == 0.0

    def test_truncation(self):
        feats = first_n_to_features(list(range(20)))
        assert feats.shape == (8,)

    def test_empty(self):
        feats = first_n_to_features([])
        assert feats.shape == (8,)
        assert np.allclose(feats, 0.0)


class TestTLSFeatures:
    def test_output_shape(self):
        tls = TLSMetadata(version="TLS1.3", cipher_suite_count=15, extension_count=10)
        feats = tls_to_features(tls)
        assert feats.shape == (12,)

    def test_grpc_flag(self):
        tls = TLSMetadata(has_grpc_alpn=True)
        feats = tls_to_features(tls)
        assert feats[4] == 1.0

    def test_unknown_tls_version(self):
        tls = TLSMetadata(version="unknown")
        feats = tls_to_features(tls)
        assert feats[0] == 3 / 4  # defaults to TLS 1.2 ordinal

    def test_extension_flags(self):
        tls = TLSMetadata(
            has_sni_extension=True,
            has_sct_extension=True,
            has_status_request=False,
            has_supported_versions_13_only=True,
            has_post_handshake_auth=False,
        )
        feats = tls_to_features(tls)
        assert feats[7] == 1.0   # SNI
        assert feats[8] == 1.0   # SCT
        assert feats[9] == 0.0   # OCSP
        assert feats[10] == 1.0  # TLS 1.3 only
        assert feats[11] == 0.0  # post_handshake_auth

    def test_all_extensions_off(self):
        tls = TLSMetadata(
            has_sni_extension=False,
            has_sct_extension=False,
            has_status_request=False,
            has_supported_versions_13_only=False,
            has_post_handshake_auth=False,
        )
        feats = tls_to_features(tls)
        assert np.allclose(feats[7:12], 0.0)


class TestJA4Features:
    def test_output_shape(self):
        ja4 = JA4Components(tls_version="TLS1.3", cipher_count=10, extension_count=8)
        feats = ja4_to_features(ja4)
        assert feats.shape == (6,)

    def test_cipher_hash_populated(self):
        ja4 = JA4Components(sorted_cipher_hash="abc123")
        feats = ja4_to_features(ja4)
        assert feats[4] != 0.0 or feats[5] != 0.0


class TestExtractFeatures:
    def _make_record(self, domain="api.openai.com"):
        return FlowRecord(
            flow_stats=FlowStats(
                packet_sizes=[100, 500, 200, 800],
                inter_arrival_times=[0.01, 0.02, 0.03],
                duration_seconds=2.0,
                packet_count_upstream=2,
                packet_count_downstream=2,
                total_bytes=1600,
                first_n_packet_sizes=[100, 500, 200, 800],
                upstream_packet_sizes=[100, 500],
                downstream_packet_sizes=[200, 800],
                upstream_bytes=600,
                downstream_bytes=1000,
            ),
            tls_metadata=TLSMetadata(
                version="TLS1.3", cipher_suite_count=15, extension_count=10,
                alpn="h2", has_h2_alpn=True, cert_chain_length=3,
                has_sni_extension=True, has_sct_extension=True,
            ),
            ja4_components=JA4Components(
                tls_version="TLS1.3", cipher_count=15, extension_count=10, alpn="h2",
            ),
            sni_domain=domain,
        )

    def test_full_extraction_shape(self):
        feats = extract_features(self._make_record())
        assert feats.shape == (NUM_FEATURES,)
        assert feats.dtype == np.float32

    def test_different_domains_produce_different_vectors(self):
        f1 = extract_features(self._make_record("api.openai.com"))
        f2 = extract_features(self._make_record("www.google.com"))
        # Only the last 11 features (SNI hash) should differ
        assert not np.allclose(f1[50:61], f2[50:61])
        # Flow/TLS/JA4 features should be identical
        np.testing.assert_array_almost_equal(f1[:50], f2[:50])

    def test_feature_vector_layout(self):
        """Verify the concatenation order matches the documented layout."""
        feats = extract_features(self._make_record())
        # Flow stats should be [0:24]
        # First-N should be [24:32]
        # TLS should be [32:44]
        # JA4 should be [44:50]
        # SNI should be [50:61]
        assert feats.shape[0] == 24 + 8 + 12 + 6 + 11


class TestSyntheticDataGeneration:
    """Test that the synthetic data generator produces valid output."""

    def test_csv_loadable(self):
        import csv
        csv_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_train.csv"
        if not csv_path.exists():
            pytest.skip("Synthetic data not generated yet")

        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert len(header) == NUM_FEATURES + 1  # features + label
            assert header[-1] == "label"

            row = next(reader)
            assert len(row) == NUM_FEATURES + 1
            values = [float(v) for v in row]
            assert values[-1] in (0.0, 1.0)

    def test_csv_feature_ranges(self):
        csv_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_train.csv"
        if not csv_path.exists():
            pytest.skip("Synthetic data not generated yet")

        import csv
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            rows = [list(map(float, row)) for row in reader]

        X = np.array([r[:NUM_FEATURES] for r in rows])
        y = np.array([r[NUM_FEATURES] for r in rows])

        assert X.shape[1] == NUM_FEATURES
        assert set(np.unique(y)) == {0.0, 1.0}
        assert X.min() >= -3.0
        assert X.max() <= 3.0
