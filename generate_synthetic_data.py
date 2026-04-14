# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Generate synthetic labeled training data for pipeline testing.

This produces realistic-ish feature vectors by simulating the statistical
properties of AI provider traffic vs. general web traffic. NOT a substitute
for real packet captures — use this only to validate the training pipeline.

Usage:
    python scripts/generate_synthetic_data.py --output data/synthetic_train.csv --samples 10000
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import FEATURE_NAMES, NUM_FEATURES

# Known AI provider SNI patterns (for generating realistic positive examples)
AI_DOMAINS = [
    "api.openai.com",
    "chat.openai.com",
    "api.anthropic.com",
    "claude.ai",
    "generativelanguage.googleapis.com",
    "gemini.google.com",
    "api.x.ai",
    "grok.x.ai",
    "api.cohere.ai",
    "api.mistral.ai",
    "api.together.xyz",
    "api.replicate.com",
    "api-inference.huggingface.co",
    "api.perplexity.ai",
    "api.deepseek.com",
]

# Common non-AI domains (negative class)
NORMAL_DOMAINS = [
    "www.google.com",
    "www.youtube.com",
    "www.facebook.com",
    "www.amazon.com",
    "cdn.jsdelivr.net",
    "fonts.googleapis.com",
    "ajax.googleapis.com",
    "api.github.com",
    "registry.npmjs.org",
    "pypi.org",
    "www.reddit.com",
    "news.ycombinator.com",
    "stackoverflow.com",
    "en.wikipedia.org",
    "docs.python.org",
    "update.googleapis.com",
    "play.googleapis.com",
    "clients2.google.com",
    "settings.data.microsoft.com",
    "login.microsoftonline.com",
]


def _sni_ngram_hash_inline(domain: str, dims: int = 3) -> np.ndarray:
    """Inline SNI hash to avoid import dependency issues in standalone script."""
    import struct as _struct

    def _mmh3(key: bytes, seed: int = 0) -> int:
        h = seed
        c1, c2, mask = 0xCC9E2D51, 0x1B873593, 0xFFFFFFFF
        for i in range(len(key) // 4):
            k = _struct.unpack_from("<I", key, i * 4)[0]
            k = (k * c1) & mask
            k = ((k << 15) | (k >> 17)) & mask
            k = (k * c2) & mask
            h ^= k
            h = ((h << 13) | (h >> 19)) & mask
            h = (h * 5 + 0xE6546B64) & mask
        tail_start = (len(key) // 4) * 4
        k1 = 0
        tl = len(key) & 3
        if tl >= 3: k1 ^= key[tail_start + 2] << 16
        if tl >= 2: k1 ^= key[tail_start + 1] << 8
        if tl >= 1:
            k1 ^= key[tail_start]
            k1 = (k1 * c1) & mask
            k1 = ((k1 << 15) | (k1 >> 17)) & mask
            k1 = (k1 * c2) & mask
            h ^= k1
        h ^= len(key)
        h ^= h >> 16; h = (h * 0x85EBCA6B) & mask
        h ^= h >> 13; h = (h * 0xC2B2AE35) & mask
        h ^= h >> 16
        return h

    vec = np.zeros(dims, dtype=np.float32)
    domain = domain.lower().strip(".")
    for n in (2, 3):
        for i in range(len(domain) - n + 1):
            gram = domain[i:i + n].encode()
            h = _mmh3(gram, seed=0)
            idx = h % dims
            sign = 1.0 if (_mmh3(gram, seed=42) & 1) == 0 else -1.0
            vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def generate_ai_traffic(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a synthetic feature vector resembling AI provider traffic.

    AI traffic characteristics:
    - Larger initial request payloads (prompt text)
    - Streaming responses with many small packets (SSE/chunked)
    - Higher downstream packet counts (streaming tokens)
    - TLS 1.3, often h2 or gRPC ALPN
    - Longer flow duration (streaming responses)
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    domain = rng.choice(AI_DOMAINS)

    # Flow statistics [0:16]
    # AI traffic: moderate request, many small response chunks
    pkt_sizes = np.concatenate([
        rng.integers(200, 1400, size=rng.integers(2, 5)),     # upstream (prompts)
        rng.integers(40, 300, size=rng.integers(20, 200)),     # downstream (token chunks)
    ])
    iats = rng.exponential(0.05, size=len(pkt_sizes))  # fast streaming

    features[0] = np.mean(pkt_sizes) / 1500
    features[1] = np.std(pkt_sizes) / 1500
    features[2] = np.min(pkt_sizes) / 1500
    features[3] = np.max(pkt_sizes) / 1500
    features[4] = np.percentile(pkt_sizes, 25) / 1500
    features[5] = np.percentile(pkt_sizes, 50) / 1500
    features[6] = np.percentile(pkt_sizes, 75) / 1500

    features[7] = np.mean(iats) / 10
    features[8] = np.std(iats) / 10
    features[9] = np.min(iats) / 10
    features[10] = np.max(iats) / 10
    features[11] = np.median(iats) / 10

    duration = float(rng.uniform(1.0, 60.0))  # streaming can be long
    features[12] = np.log1p(duration) / np.log1p(300)
    features[13] = np.log1p(rng.integers(2, 10)) / np.log1p(10000)  # upstream pkts
    features[14] = np.log1p(rng.integers(20, 500)) / np.log1p(10000)  # downstream pkts
    bps = float(np.sum(pkt_sizes)) / max(duration, 0.001)
    features[15] = np.log1p(bps) / np.log1p(1e9)

    # First-N packet sizes [16:24]
    first_n = list(pkt_sizes[:8])
    for i in range(min(len(first_n), 8)):
        features[16 + i] = first_n[i] / 1500

    # TLS metadata [24:31]
    features[24] = 4 / 4  # TLS 1.3
    features[25] = rng.uniform(0.3, 0.8)  # cipher count
    features[26] = rng.uniform(0.3, 0.7)  # extension count
    has_grpc = rng.random() < 0.3  # 30% chance of gRPC
    features[27] = (5 if has_grpc else 3) / 5  # ALPN (grpc or h2)
    features[28] = 1.0 if has_grpc else 0.0
    features[29] = 1.0  # h2 almost always
    features[30] = rng.uniform(0.4, 0.8)  # cert chain length

    # JA4 components [31:37]
    features[31] = 4 / 4
    features[32] = rng.uniform(0.3, 0.7)
    features[33] = rng.uniform(0.3, 0.6)
    features[34] = features[27]
    features[35] = rng.random()  # cipher hash dim 0
    features[36] = rng.random()  # cipher hash dim 1

    # SNI n-gram hash [37:40]
    sni_hash = _sni_ngram_hash_inline(domain)
    features[37:40] = sni_hash

    # Add noise
    features += rng.normal(0, 0.02, size=NUM_FEATURES).astype(np.float32)
    return np.clip(features, -2.0, 2.0)


def generate_normal_traffic(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a synthetic feature vector resembling normal web traffic.

    Normal traffic characteristics:
    - Typical request/response pattern (fewer, larger responses)
    - Shorter flow durations (page loads complete quickly)
    - More uniform packet sizes
    - Mixed TLS versions, often h2 or http/1.1
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    domain = rng.choice(NORMAL_DOMAINS)

    # Flow statistics [0:16]
    pkt_sizes = np.concatenate([
        rng.integers(100, 600, size=rng.integers(1, 5)),       # requests
        rng.integers(500, 1460, size=rng.integers(5, 50)),     # responses (larger chunks)
    ])
    iats = rng.exponential(0.2, size=len(pkt_sizes))  # slower than streaming

    features[0] = np.mean(pkt_sizes) / 1500
    features[1] = np.std(pkt_sizes) / 1500
    features[2] = np.min(pkt_sizes) / 1500
    features[3] = np.max(pkt_sizes) / 1500
    features[4] = np.percentile(pkt_sizes, 25) / 1500
    features[5] = np.percentile(pkt_sizes, 50) / 1500
    features[6] = np.percentile(pkt_sizes, 75) / 1500

    features[7] = np.mean(iats) / 10
    features[8] = np.std(iats) / 10
    features[9] = np.min(iats) / 10
    features[10] = np.max(iats) / 10
    features[11] = np.median(iats) / 10

    duration = float(rng.uniform(0.1, 10.0))  # typically shorter
    features[12] = np.log1p(duration) / np.log1p(300)
    features[13] = np.log1p(rng.integers(1, 10)) / np.log1p(10000)
    features[14] = np.log1p(rng.integers(5, 100)) / np.log1p(10000)
    bps = float(np.sum(pkt_sizes)) / max(duration, 0.001)
    features[15] = np.log1p(bps) / np.log1p(1e9)

    # First-N packet sizes [16:24]
    first_n = list(pkt_sizes[:8])
    for i in range(min(len(first_n), 8)):
        features[16 + i] = first_n[i] / 1500

    # TLS metadata [24:31]
    tls_ver = rng.choice([3, 4], p=[0.3, 0.7])  # mix of 1.2 and 1.3
    features[24] = tls_ver / 4
    features[25] = rng.uniform(0.2, 0.9)
    features[26] = rng.uniform(0.2, 0.8)
    features[27] = rng.choice([2, 3]) / 5  # http/1.1 or h2
    features[28] = 0.0  # no gRPC
    features[29] = 1.0 if rng.random() < 0.7 else 0.0  # h2 common
    features[30] = rng.uniform(0.2, 0.8)

    # JA4 components [31:37]
    features[31] = tls_ver / 4
    features[32] = rng.uniform(0.2, 0.8)
    features[33] = rng.uniform(0.2, 0.7)
    features[34] = features[27]
    features[35] = rng.random()
    features[36] = rng.random()

    # SNI n-gram hash [37:40]
    sni_hash = _sni_ngram_hash_inline(domain)
    features[37:40] = sni_hash

    # Add noise
    features += rng.normal(0, 0.02, size=NUM_FEATURES).astype(np.float32)
    return np.clip(features, -2.0, 2.0)


def generate_dataset(
    n_samples: int,
    ai_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced-ish synthetic dataset.

    Args:
        n_samples: Total number of samples
        ai_ratio: Fraction of positive (AI traffic) samples. Default 0.15 (realistic).
        seed: Random seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, 40)
        y: Labels (n_samples,) — 1.0 for AI traffic, 0.0 for normal
    """
    rng = np.random.default_rng(seed)

    n_ai = int(n_samples * ai_ratio)
    n_normal = n_samples - n_ai

    X_ai = np.stack([generate_ai_traffic(rng) for _ in range(n_ai)])
    X_normal = np.stack([generate_normal_traffic(rng) for _ in range(n_normal)])

    X = np.vstack([X_ai, X_normal])
    y = np.concatenate([np.ones(n_ai), np.zeros(n_normal)])

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--ai-ratio", type=float, default=0.15, help="Fraction of AI traffic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating {args.samples} samples (AI ratio: {args.ai_ratio})...")
    X, y = generate_dataset(args.samples, ai_ratio=args.ai_ratio, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label"])
        for i in range(len(X)):
            writer.writerow([f"{v:.6f}" for v in X[i]] + [f"{y[i]:.0f}"])

    n_ai = int(y.sum())
    print(f"Saved to {output_path}")
    print(f"  Total: {len(y)}, AI: {n_ai} ({n_ai/len(y)*100:.1f}%), Normal: {len(y)-n_ai}")


if __name__ == "__main__":
    main()
