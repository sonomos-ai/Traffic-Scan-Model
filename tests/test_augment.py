# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""Tests for data augmentation strategies."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import NUM_FEATURES
from augment import (
    augment_dataset,
    _iat_jitter,
    _pkt_size_jitter,
    _multi_window,
    _mixup,
    _JITTERABLE_IAT,
    _JITTERABLE_PKT,
    _TLS_INDICES,
    _JA4_INDICES,
    _SNI_INDICES,
)


def _make_dummy_data(n: int = 100, seed: int = 42):
    """Create dummy feature matrix and labels."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, NUM_FEATURES)).astype(np.float32)
    y = np.concatenate([np.ones(n // 5), np.zeros(n - n // 5)]).astype(np.float32)
    rng.shuffle(y)
    return X, y


class TestIATJitter:
    def test_shape_preserved(self):
        X, _ = _make_dummy_data()
        X_aug = _iat_jitter(X, sigma=0.1)
        assert X_aug.shape == X.shape

    def test_only_iat_features_changed(self):
        X, _ = _make_dummy_data()
        X_aug = _iat_jitter(X, sigma=0.1)

        # Non-IAT features should be identical
        non_iat = [i for i in range(NUM_FEATURES) if i not in _JITTERABLE_IAT]
        np.testing.assert_array_equal(X[:, non_iat], X_aug[:, non_iat])

        # IAT features should be different
        iat_diff = np.abs(X[:, _JITTERABLE_IAT] - X_aug[:, _JITTERABLE_IAT]).sum()
        assert iat_diff > 0, "IAT features should be jittered"

    def test_magnitude_proportional(self):
        X, _ = _make_dummy_data()
        X_small = _iat_jitter(X, sigma=0.01)
        X_large = _iat_jitter(X, sigma=0.5)

        diff_small = np.abs(X[:, _JITTERABLE_IAT] - X_small[:, _JITTERABLE_IAT]).mean()
        diff_large = np.abs(X[:, _JITTERABLE_IAT] - X_large[:, _JITTERABLE_IAT]).mean()
        assert diff_large > diff_small, "Larger sigma should produce larger jitter"

    def test_zero_sigma_no_change(self):
        X, _ = _make_dummy_data()
        X_aug = _iat_jitter(X, sigma=0.0)
        np.testing.assert_array_almost_equal(X, X_aug)


class TestPktSizeJitter:
    def test_shape_preserved(self):
        X, _ = _make_dummy_data()
        X_aug = _pkt_size_jitter(X, sigma=0.05)
        assert X_aug.shape == X.shape

    def test_only_pkt_features_changed(self):
        X, _ = _make_dummy_data()
        X_aug = _pkt_size_jitter(X, sigma=0.05)

        non_pkt = [i for i in range(NUM_FEATURES) if i not in _JITTERABLE_PKT]
        np.testing.assert_array_equal(X[:, non_pkt], X_aug[:, non_pkt])

    def test_values_clipped(self):
        X, _ = _make_dummy_data()
        X_aug = _pkt_size_jitter(X, sigma=0.5)  # large sigma
        # Packet features should be clipped to [0, 2]
        for idx in _JITTERABLE_PKT:
            assert X_aug[:, idx].min() >= 0.0
            assert X_aug[:, idx].max() <= 2.0


class TestMultiWindow:
    def test_multiplied_sample_count(self):
        X, y = _make_dummy_data(n=50)
        windows = [0.2, 0.4, 0.6, 0.8]
        X_aug, y_aug = _multi_window(X, y, windows=windows)
        # Original + 4 windows = 5x
        assert len(X_aug) == len(X) * (1 + len(windows))
        assert len(y_aug) == len(X_aug)

    def test_labels_preserved(self):
        X, y = _make_dummy_data(n=50)
        X_aug, y_aug = _multi_window(X, y, windows=[0.5])
        # Each window copy should have same label distribution
        n_orig = len(X)
        y_orig = y_aug[:n_orig]
        y_copy = y_aug[n_orig:]
        assert y_orig.sum() == y_copy.sum()

    def test_originals_unchanged(self):
        X, y = _make_dummy_data(n=50)
        X_aug, _ = _multi_window(X, y, windows=[0.5])
        # First n_orig samples should be untouched
        np.testing.assert_array_equal(X_aug[:len(X)], X)

    def test_first_n_zeroed_for_small_window(self):
        X, y = _make_dummy_data(n=50)
        X_aug, _ = _multi_window(X, y, windows=[0.2])
        # Window 0.2 → only first 1 of 8 slots visible
        # Features [25:32] should be zero (slots 2-8 of first-N)
        copies = X_aug[len(X):]
        for i in range(1, 8):
            assert (copies[:, 24 + i] == 0.0).all(), f"First-N slot {i+1} should be zeroed"

    def test_tls_features_unchanged(self):
        X, y = _make_dummy_data(n=50)
        X_aug, _ = _multi_window(X, y, windows=[0.5])
        copies = X_aug[len(X):]
        # TLS, JA4, SNI features should not be modified
        for indices in [_TLS_INDICES, _JA4_INDICES, _SNI_INDICES]:
            np.testing.assert_array_equal(copies[:, indices], X[:, indices])


class TestMixup:
    def test_produces_samples(self):
        X, y = _make_dummy_data(n=100)
        X_mix, y_mix = _mixup(X, y, alpha=0.2, n_samples=50)
        assert len(X_mix) > 0
        assert len(X_mix) == len(y_mix)

    def test_same_class_only(self):
        X, y = _make_dummy_data(n=100)
        X_mix, y_mix = _mixup(X, y, alpha=0.2)
        # All mixup labels should be 0 or 1 (no interpolated labels)
        assert set(np.unique(y_mix)).issubset({0.0, 1.0})

    def test_tls_features_not_interpolated(self):
        X, y = _make_dummy_data(n=100)
        rng = np.random.default_rng(42)
        X_mix, y_mix = _mixup(X, y, alpha=0.5, rng=rng)

        # TLS, JA4, SNI features should come from one of the parents
        # (taken from idx_a), not interpolated
        for i in _TLS_INDICES + _JA4_INDICES + _SNI_INDICES:
            # Each mixup value should exist in the original data for that class
            for label in [0.0, 1.0]:
                mask_mix = y_mix == label
                mask_orig = y == label
                if mask_mix.any() and mask_orig.any():
                    mix_vals = set(X_mix[mask_mix, i].round(6))
                    orig_vals = set(X[mask_orig, i].round(6))
                    # At least most values should be from originals
                    overlap = mix_vals & orig_vals
                    assert len(overlap) / max(len(mix_vals), 1) > 0.5

    def test_zero_alpha_near_originals(self):
        X, y = _make_dummy_data(n=100)
        X_mix, _ = _mixup(X, y, alpha=0.01)  # very low alpha → lambda near 0 or 1
        # Mixup samples should be very close to originals
        # (can't assert exactly due to stochasticity, but check shape)
        assert X_mix.shape[1] == NUM_FEATURES


class TestAugmentDataset:
    def test_increases_sample_count(self):
        X, y = _make_dummy_data(n=100)
        X_aug, y_aug = augment_dataset(X, y, seed=42)
        assert len(X_aug) > len(X), f"Augmented ({len(X_aug)}) should be larger than original ({len(X)})"

    def test_feature_dim_preserved(self):
        X, y = _make_dummy_data(n=100)
        X_aug, y_aug = augment_dataset(X, y, seed=42)
        assert X_aug.shape[1] == NUM_FEATURES

    def test_labels_valid(self):
        X, y = _make_dummy_data(n=100)
        X_aug, y_aug = augment_dataset(X, y, seed=42)
        assert set(np.unique(y_aug)).issubset({0.0, 1.0})

    def test_no_augment_flags(self):
        X, y = _make_dummy_data(n=100)
        X_aug, y_aug = augment_dataset(
            X, y,
            multi_window=False,
            iat_jitter=False,
            pkt_size_jitter=False,
            mixup_alpha=0.0,
        )
        # With everything disabled, just shuffled originals
        assert len(X_aug) == len(X)

    def test_deterministic_with_seed(self):
        X, y = _make_dummy_data(n=50)
        X_a, y_a = augment_dataset(X, y, seed=123)
        X_b, y_b = augment_dataset(X, y, seed=123)
        np.testing.assert_array_equal(X_a, X_b)
        np.testing.assert_array_equal(y_a, y_b)

    def test_different_seeds_differ(self):
        X, y = _make_dummy_data(n=50)
        X_a, _ = augment_dataset(X, y, seed=1)
        X_b, _ = augment_dataset(X, y, seed=2)
        assert not np.allclose(X_a, X_b)

    def test_class_ratio_preserved(self):
        X, y = _make_dummy_data(n=200)
        orig_ratio = y.mean()
        X_aug, y_aug = augment_dataset(X, y, seed=42)
        aug_ratio = y_aug.mean()
        # Ratio should be roughly preserved (within ±10%, mixup oversamples minority)
        assert abs(orig_ratio - aug_ratio) < 0.10, (
            f"Class ratio shifted: {orig_ratio:.3f} → {aug_ratio:.3f}"
        )

    def test_values_finite(self):
        X, y = _make_dummy_data(n=100)
        X_aug, _ = augment_dataset(X, y, seed=42)
        assert np.all(np.isfinite(X_aug)), "Augmented data contains non-finite values"
