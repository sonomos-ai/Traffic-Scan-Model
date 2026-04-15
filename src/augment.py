# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Data augmentation for the Sonomos Traffic Classifier.

Augmentation strategies for flow-level traffic features:

1. Multi-window extraction: extract features from the first N packets of a
   flow (N = 5, 10, 20, 50, all) as separate training samples. Simulates the
   classifier running at different points during a connection's lifecycle.

2. IAT jitter: add ±10% Gaussian noise to inter-arrival time features.
   Simulates network condition variation (congestion, different paths).

3. Packet size perturbation: add ±5% noise to packet size features.
   Simulates MTU differences, TCP segmentation offload variation.

4. Feature-space mixup: convex interpolation between same-class samples.
   Regularization technique that smooths the decision boundary.

Usage:
    from augment import augment_dataset

    X_aug, y_aug = augment_dataset(
        X_train, y_train,
        multi_window=True,
        iat_jitter=True,
        pkt_size_jitter=True,
        mixup_alpha=0.2,
        seed=42,
    )
"""

import numpy as np
from typing import Optional


# Feature index ranges (must match features.py layout)
_PKT_SIZE_INDICES = list(range(0, 7))       # [0:7] packet size stats
_IAT_INDICES = list(range(7, 12))           # [7:12] IAT stats
_DURATION_INDEX = 12
_PKT_COUNT_UP_INDEX = 13
_PKT_COUNT_DOWN_INDEX = 14
_BPS_INDEX = 15
_UP_PKT_INDICES = list(range(16, 19))      # [16:19] upstream pkt size stats
_DOWN_PKT_INDICES = list(range(19, 22))     # [19:22] downstream pkt size stats
_BYTE_RATIO_INDEX = 22
_PKT_RATIO_INDEX = 23
_FIRST_N_INDICES = list(range(24, 32))      # [24:32] first-N packet sizes
_TLS_INDICES = list(range(32, 44))          # [32:44] TLS metadata (don't augment)
_JA4_INDICES = list(range(44, 50))          # [44:50] JA4 components (don't augment)
_SNI_INDICES = list(range(50, 61))          # [50:61] SNI hash (don't augment)

# Indices that are safe to apply continuous noise to
_JITTERABLE_PKT = _PKT_SIZE_INDICES + _UP_PKT_INDICES + _DOWN_PKT_INDICES + _FIRST_N_INDICES
_JITTERABLE_IAT = _IAT_INDICES


def _iat_jitter(
    X: np.ndarray,
    sigma: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add Gaussian noise to IAT features (±sigma relative).

    Network conditions vary — congestion, route changes, queuing delays
    all affect inter-arrival times without changing the application behavior.
    This teaches the model to be robust to timing noise.

    Args:
        X: Feature matrix (n_samples, 61)
        sigma: Relative noise std (0.1 = ±10%)
        rng: Random generator
    """
    if rng is None:
        rng = np.random.default_rng()

    X_aug = X.copy()
    for idx in _JITTERABLE_IAT:
        noise = 1.0 + rng.normal(0, sigma, size=X.shape[0]).astype(np.float32)
        noise = np.clip(noise, 1.0 - 3 * sigma, 1.0 + 3 * sigma)  # clip outliers
        X_aug[:, idx] *= noise

    return X_aug


def _pkt_size_jitter(
    X: np.ndarray,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add Gaussian noise to packet size features (±sigma relative).

    TCP segmentation offload, MTU differences, and retransmission
    behavior cause minor packet size variation for the same application.

    Args:
        X: Feature matrix (n_samples, 61)
        sigma: Relative noise std (0.05 = ±5%)
        rng: Random generator
    """
    if rng is None:
        rng = np.random.default_rng()

    X_aug = X.copy()
    for idx in _JITTERABLE_PKT:
        noise = 1.0 + rng.normal(0, sigma, size=X.shape[0]).astype(np.float32)
        noise = np.clip(noise, 1.0 - 3 * sigma, 1.0 + 3 * sigma)
        X_aug[:, idx] *= noise
        X_aug[:, idx] = np.clip(X_aug[:, idx], 0.0, 2.0)  # keep in valid range

    return X_aug


def _multi_window(
    X: np.ndarray,
    y: np.ndarray,
    windows: list[float] = [0.2, 0.4, 0.6, 0.8],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate observing a flow at different points in its lifecycle.

    For each sample, create additional samples that represent seeing only
    a fraction of the flow's packets. This is done by scaling down:
    - packet counts (log-scaled, so we adjust the raw count before re-logging)
    - duration
    - throughput
    - first-N packet sizes (zero out later slots)
    - byte counts and ratios stay proportional

    This teaches the model to classify flows from partial observations,
    which is critical since the classifier may run early in a connection.

    Args:
        X: Feature matrix (n_samples, 61)
        y: Labels (n_samples,)
        windows: Fractions of the flow to simulate (0.2 = first 20% of packets)

    Returns:
        X_aug: Augmented feature matrix including originals
        y_aug: Labels for augmented samples
    """
    all_X = [X]
    all_y = [y]

    for frac in windows:
        X_w = X.copy()

        # Scale duration down proportionally
        X_w[:, _DURATION_INDEX] *= frac

        # Scale packet counts: these are log1p(count)/log1p(10000)
        # Undo log-normalization, scale, re-normalize
        log_norm = np.log1p(10000)
        for idx in [_PKT_COUNT_UP_INDEX, _PKT_COUNT_DOWN_INDEX]:
            raw = np.expm1(X_w[:, idx] * log_norm)
            raw = np.maximum(raw * frac, 1.0)
            X_w[:, idx] = np.log1p(raw) / log_norm

        # Scale throughput: log1p(bps)/log1p(1e9)
        # Throughput ~ total_bytes / duration, both scale by frac → stays similar
        # But fewer packets = less total bytes, so reduce slightly
        X_w[:, _BPS_INDEX] *= (0.9 + 0.1 * frac)

        # Zero out later first-N packet slots
        n_visible = max(1, int(8 * frac))
        for i in range(n_visible, 8):
            X_w[:, _FIRST_N_INDICES[i]] = 0.0

        # Directional stats: scale proportionally (already normalized)
        # Ratios stay the same (they're ratios, not absolutes)

        all_X.append(X_w)
        all_y.append(y.copy())

    return np.vstack(all_X), np.concatenate(all_y)


def _mixup(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.2,
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Feature-space mixup between same-class samples.

    Creates synthetic training examples by convex interpolation:
        x' = lambda * x_i + (1 - lambda) * x_j
        y' = y_i  (same class, so label is unchanged)

    Only mixes within the same class to avoid creating ambiguous
    boundary samples. Uses Beta(alpha, alpha) for lambda.

    Only applied to flow/packet features [0:32]. TLS, JA4, and SNI
    features are taken from the first sample (these are categorical
    or hash-based and don't interpolate meaningfully).

    Args:
        X: Feature matrix (n_samples, 61)
        y: Labels (n_samples,)
        alpha: Beta distribution parameter (lower = closer to originals)
        n_samples: Number of mixup samples to generate (default: len(X) // 2)
        rng: Random generator

    Returns:
        X_mixed: Mixup samples only (not including originals)
        y_mixed: Labels for mixup samples
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_samples is None:
        n_samples = len(X) // 2

    X_mixed = []
    y_mixed = []

    # Flow/packet feature indices that interpolate meaningfully
    interp_indices = list(range(0, 32))

    for label in [0.0, 1.0]:
        mask = y == label
        X_class = X[mask]
        if len(X_class) < 2:
            continue

        n_class = min(n_samples // 2, len(X_class))
        idx_a = rng.integers(0, len(X_class), size=n_class)
        idx_b = rng.integers(0, len(X_class), size=n_class)

        lam = rng.beta(alpha, alpha, size=(n_class, 1)).astype(np.float32)

        # Interpolate flow features
        X_new = X_class[idx_a].copy()
        for i in interp_indices:
            X_new[:, i] = lam[:, 0] * X_class[idx_a, i] + (1 - lam[:, 0]) * X_class[idx_b, i]

        X_mixed.append(X_new)
        y_mixed.append(np.full(n_class, label, dtype=np.float32))

    if not X_mixed:
        return np.empty((0, X.shape[1])), np.empty(0)

    return np.vstack(X_mixed), np.concatenate(y_mixed)


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    multi_window: bool = True,
    iat_jitter: bool = True,
    pkt_size_jitter: bool = True,
    mixup_alpha: float = 0.2,
    n_mixup: int | None = None,
    iat_sigma: float = 0.1,
    pkt_sigma: float = 0.05,
    windows: list[float] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply all augmentation strategies to a training dataset.

    Augmentation order:
    1. Multi-window extraction (creates N copies at different observation points)
    2. IAT jitter on all samples (including windowed copies)
    3. Packet size jitter on all samples
    4. Mixup between same-class samples

    Validation data should NOT be augmented — only training data.

    Args:
        X: Feature matrix (n_samples, 61)
        y: Labels (n_samples,)
        multi_window: Enable multi-window extraction
        iat_jitter: Enable IAT noise
        pkt_size_jitter: Enable packet size noise
        mixup_alpha: Beta distribution parameter for mixup (0 = disabled)
        n_mixup: Number of mixup samples (default: len(X) // 2)
        iat_sigma: IAT noise std (relative)
        pkt_sigma: Packet size noise std (relative)
        windows: Window fractions for multi-window (default: [0.2, 0.4, 0.6, 0.8])
        seed: Random seed

    Returns:
        X_aug: Augmented feature matrix
        y_aug: Augmented labels
    """
    rng = np.random.default_rng(seed)
    n_orig = len(X)

    if windows is None:
        windows = [0.2, 0.4, 0.6, 0.8]

    # 1. Multi-window
    if multi_window:
        X, y = _multi_window(X, y, windows=windows)
        print(f"  Multi-window: {n_orig} → {len(X)} samples "
              f"({len(windows)} windows + original)")

    # 2. IAT jitter (applied to copies, not originals)
    if iat_jitter and len(X) > n_orig:
        # Only jitter the augmented copies, not the originals
        X_orig = X[:n_orig]
        X_copies = X[n_orig:]
        X_copies = _iat_jitter(X_copies, sigma=iat_sigma, rng=rng)
        X = np.vstack([X_orig, X_copies])
        print(f"  IAT jitter: σ={iat_sigma} applied to {len(X_copies)} copies")
    elif iat_jitter:
        # No multi-window, jitter everything
        X_jittered = _iat_jitter(X, sigma=iat_sigma, rng=rng)
        X = np.vstack([X, X_jittered])
        y = np.concatenate([y, y])
        print(f"  IAT jitter: σ={iat_sigma}, doubled to {len(X)} samples")

    # 3. Packet size jitter
    if pkt_size_jitter and len(X) > n_orig:
        X_orig = X[:n_orig]
        X_copies = X[n_orig:]
        X_copies = _pkt_size_jitter(X_copies, sigma=pkt_sigma, rng=rng)
        X = np.vstack([X_orig, X_copies])
        print(f"  Pkt size jitter: σ={pkt_sigma} applied to {len(X_copies)} copies")
    elif pkt_size_jitter:
        X_jittered = _pkt_size_jitter(X, sigma=pkt_sigma, rng=rng)
        X = np.vstack([X, X_jittered])
        y = np.concatenate([y, y])
        print(f"  Pkt size jitter: σ={pkt_sigma}, doubled to {len(X)} samples")

    # 4. Mixup
    if mixup_alpha > 0:
        X_mix, y_mix = _mixup(X, y, alpha=mixup_alpha, n_samples=n_mixup, rng=rng)
        if len(X_mix) > 0:
            X = np.vstack([X, X_mix])
            y = np.concatenate([y, y_mix])
            print(f"  Mixup: α={mixup_alpha}, added {len(X_mix)} samples → {len(X)} total")

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]
