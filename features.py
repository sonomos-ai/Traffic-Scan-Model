# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Feature extraction for the Sonomos Traffic Classifier.

Converts raw flow metadata into a 40-dimension numeric feature vector:
  [0:16]  Flow statistics (packet sizes, IATs, duration, counts, throughput)
  [16:24] First-N packet sizes (first 8 packets, upstream/downstream interleaved)
  [24:31] TLS handshake metadata
  [31:37] JA4 fingerprint components
  [37:40] SNI character n-gram hash
"""

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


NUM_FEATURES = 40

# --- SNI n-gram hashing ---

_HASH_DIMS = 3  # dimensions for SNI hash vector


def _murmurhash3_32(key: bytes, seed: int = 0) -> int:
    """Simplified MurmurHash3 32-bit for deterministic cross-platform hashing."""
    h = seed
    length = len(key)
    nblocks = length // 4

    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    mask32 = 0xFFFFFFFF

    for i in range(nblocks):
        k = struct.unpack_from("<I", key, i * 4)[0]
        k = (k * c1) & mask32
        k = ((k << 15) | (k >> 17)) & mask32
        k = (k * c2) & mask32

        h ^= k
        h = ((h << 13) | (h >> 19)) & mask32
        h = (h * 5 + 0xE6546B64) & mask32

    tail_start = nblocks * 4
    k1 = 0
    tail_len = length & 3
    if tail_len >= 3:
        k1 ^= key[tail_start + 2] << 16
    if tail_len >= 2:
        k1 ^= key[tail_start + 1] << 8
    if tail_len >= 1:
        k1 ^= key[tail_start]
        k1 = (k1 * c1) & mask32
        k1 = ((k1 << 15) | (k1 >> 17)) & mask32
        k1 = (k1 * c2) & mask32
        h ^= k1

    h ^= length
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & mask32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & mask32
    h ^= h >> 16

    return h


def sni_ngram_hash(domain: str, dims: int = _HASH_DIMS) -> np.ndarray:
    """
    Hash character 2-grams and 3-grams of an SNI domain into a fixed-size vector.

    Uses the hashing trick (Weinberger et al., 2009): each n-gram is hashed
    with MurmurHash3, mapped to a dimension via modulo, and accumulated with
    a sign hash for variance reduction.

    Args:
        domain: SNI hostname (e.g., "api.openai.com")
        dims: Output vector dimensionality (default 3)

    Returns:
        np.ndarray of shape (dims,), L2-normalized
    """
    vec = np.zeros(dims, dtype=np.float32)
    domain = domain.lower().strip(".")

    ngrams = []
    for n in (2, 3):
        for i in range(len(domain) - n + 1):
            ngrams.append(domain[i : i + n])

    for gram in ngrams:
        h = _murmurhash3_32(gram.encode("utf-8"), seed=0)
        idx = h % dims
        # Sign hash for variance reduction
        sign = 1.0 if (_murmurhash3_32(gram.encode("utf-8"), seed=42) & 1) == 0 else -1.0
        vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# --- JA4 decomposition ---

# Ordinal encoding for TLS versions
_TLS_VERSION_ORD = {
    "SSLv3": 0,
    "TLSv1.0": 1,
    "TLS1.0": 1,
    "TLSv1.1": 2,
    "TLS1.1": 2,
    "TLSv1.2": 3,
    "TLS1.2": 3,
    "TLSv1.3": 4,
    "TLS1.3": 4,
}

# Ordinal encoding for ALPN values
_ALPN_ORD = {
    "": 0,
    "http/1.0": 1,
    "http/1.1": 2,
    "h2": 3,
    "h3": 4,
    "grpc": 5,
}


@dataclass
class JA4Components:
    """Decomposed JA4 fingerprint for numeric feature encoding."""

    tls_version: str = "TLS1.3"
    cipher_count: int = 0
    extension_count: int = 0
    alpn: str = ""
    sorted_cipher_hash: str = ""
    sorted_extension_hash: str = ""


def ja4_to_features(ja4: JA4Components) -> np.ndarray:
    """
    Convert decomposed JA4 components to 6-dim numeric vector.

    [0] TLS version ordinal (0-4)
    [1] Cipher suite count (normalized by /30)
    [2] Extension count (normalized by /30)
    [3] ALPN ordinal (0-5)
    [4-5] Sorted cipher hash → 2-dim via hashing trick
    """
    features = np.zeros(6, dtype=np.float32)
    features[0] = _TLS_VERSION_ORD.get(ja4.tls_version, 3) / 4.0
    features[1] = min(ja4.cipher_count / 30.0, 1.0)
    features[2] = min(ja4.extension_count / 30.0, 1.0)
    features[3] = _ALPN_ORD.get(ja4.alpn, 0) / 5.0

    # Hash the sorted cipher/ext strings into low-dim vectors
    if ja4.sorted_cipher_hash:
        h = _murmurhash3_32(ja4.sorted_cipher_hash.encode(), seed=100)
        features[4] = (h & 0xFFFF) / 65535.0
        features[5] = ((h >> 16) & 0xFFFF) / 65535.0

    return features


# --- TLS metadata encoding ---


@dataclass
class TLSMetadata:
    """Raw TLS handshake metadata extracted from ClientHello / ServerHello."""

    version: str = "TLS1.3"
    cipher_suite_count: int = 0
    extension_count: int = 0
    alpn: str = ""
    has_grpc_alpn: bool = False
    has_h2_alpn: bool = False
    cert_chain_length: int = 0


def tls_to_features(tls: TLSMetadata) -> np.ndarray:
    """
    Convert TLS metadata to 7-dim numeric vector.

    [0] TLS version ordinal (normalized)
    [1] Cipher suite count (normalized)
    [2] Extension count (normalized)
    [3] ALPN ordinal (normalized)
    [4] Has gRPC ALPN (binary)
    [5] Has H2 ALPN (binary)
    [6] Certificate chain length (normalized)
    """
    features = np.zeros(7, dtype=np.float32)
    features[0] = _TLS_VERSION_ORD.get(tls.version, 3) / 4.0
    features[1] = min(tls.cipher_suite_count / 30.0, 1.0)
    features[2] = min(tls.extension_count / 30.0, 1.0)
    features[3] = _ALPN_ORD.get(tls.alpn, 0) / 5.0
    features[4] = 1.0 if tls.has_grpc_alpn else 0.0
    features[5] = 1.0 if tls.has_h2_alpn else 0.0
    features[6] = min(tls.cert_chain_length / 5.0, 1.0)
    return features


# --- Flow statistics ---


@dataclass
class FlowStats:
    """Aggregated per-flow statistics from packet capture."""

    packet_sizes: list[int] = field(default_factory=list)
    inter_arrival_times: list[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    packet_count_upstream: int = 0
    packet_count_downstream: int = 0
    total_bytes: int = 0
    first_n_packet_sizes: list[int] = field(default_factory=list)  # first 8 packets


def _percentile_stats(values: list[float]) -> tuple[float, float, float, float, float, float, float]:
    """Compute mean, std, min, max, p25, p50, p75 from a list of values."""
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.array(values, dtype=np.float32)
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
    )


def flow_to_features(flow: FlowStats) -> np.ndarray:
    """
    Convert flow statistics to 16-dim numeric vector.

    [0:7]  Packet size stats (mean/std/min/max/p25/p50/p75), normalized by /1500
    [7:12] IAT stats (mean/std/min/max/p50), normalized by /1.0 (seconds)
    [12]   Flow duration (log-scaled)
    [13]   Upstream packet count (log-scaled)
    [14]   Downstream packet count (log-scaled)
    [15]   Bytes per second (log-scaled)
    """
    features = np.zeros(16, dtype=np.float32)

    # Packet size statistics (normalized by typical MTU)
    pkt_stats = _percentile_stats([float(s) for s in flow.packet_sizes])
    for i in range(7):
        features[i] = pkt_stats[i] / 1500.0

    # Inter-arrival time statistics
    iat_stats = _percentile_stats(flow.inter_arrival_times)
    for i, j in enumerate([0, 1, 2, 3, 5]):  # mean, std, min, max, p50
        features[7 + i] = min(iat_stats[j], 10.0) / 10.0  # cap at 10s

    # Duration (log-scaled, cap at 300s)
    features[12] = math.log1p(min(flow.duration_seconds, 300.0)) / math.log1p(300.0)

    # Packet counts (log-scaled)
    features[13] = math.log1p(flow.packet_count_upstream) / math.log1p(10000)
    features[14] = math.log1p(flow.packet_count_downstream) / math.log1p(10000)

    # Throughput (log-scaled)
    bps = flow.total_bytes / max(flow.duration_seconds, 0.001)
    features[15] = math.log1p(bps) / math.log1p(1e9)  # normalize against 1Gbps

    return features


def first_n_to_features(first_n_sizes: list[int], n: int = 8) -> np.ndarray:
    """
    Encode first N packet sizes as feature vector.

    Pads with zeros if fewer than N packets observed.
    Normalized by /1500 (typical MTU).
    """
    features = np.zeros(n, dtype=np.float32)
    for i in range(min(len(first_n_sizes), n)):
        features[i] = first_n_sizes[i] / 1500.0
    return features


# --- Combined feature vector ---


@dataclass
class FlowRecord:
    """Complete flow record combining all metadata for feature extraction."""

    flow_stats: FlowStats
    tls_metadata: TLSMetadata
    ja4_components: JA4Components
    sni_domain: str = ""


def extract_features(record: FlowRecord) -> np.ndarray:
    """
    Extract the full 40-dim feature vector from a flow record.

    Returns np.ndarray of shape (40,) with all features normalized to ~[0, 1].
    """
    flow_feats = flow_to_features(record.flow_stats)           # 16 dims
    first_n_feats = first_n_to_features(
        record.flow_stats.first_n_packet_sizes, n=8
    )                                                           # 8 dims
    tls_feats = tls_to_features(record.tls_metadata)            # 7 dims
    ja4_feats = ja4_to_features(record.ja4_components)          # 6 dims
    sni_feats = sni_ngram_hash(record.sni_domain, dims=3)       # 3 dims

    combined = np.concatenate([
        flow_feats,     # [0:16]
        first_n_feats,  # [16:24]
        tls_feats,      # [24:31]
        ja4_feats,      # [31:37]
        sni_feats,      # [37:40]
    ])

    assert combined.shape == (NUM_FEATURES,), f"Expected {NUM_FEATURES} features, got {combined.shape[0]}"
    return combined


# Feature names for debugging and analysis
FEATURE_NAMES = [
    # Flow statistics [0:16]
    "pkt_size_mean", "pkt_size_std", "pkt_size_min", "pkt_size_max",
    "pkt_size_p25", "pkt_size_p50", "pkt_size_p75",
    "iat_mean", "iat_std", "iat_min", "iat_max", "iat_p50",
    "duration", "pkt_count_up", "pkt_count_down", "bytes_per_sec",
    # First-N packet sizes [16:24]
    "first_pkt_1", "first_pkt_2", "first_pkt_3", "first_pkt_4",
    "first_pkt_5", "first_pkt_6", "first_pkt_7", "first_pkt_8",
    # TLS metadata [24:31]
    "tls_version", "tls_cipher_count", "tls_ext_count", "tls_alpn",
    "tls_has_grpc", "tls_has_h2", "tls_cert_chain_len",
    # JA4 components [31:37]
    "ja4_tls_ver", "ja4_cipher_count", "ja4_ext_count", "ja4_alpn",
    "ja4_cipher_hash_0", "ja4_cipher_hash_1",
    # SNI n-gram hash [37:40]
    "sni_hash_0", "sni_hash_1", "sni_hash_2",
]

assert len(FEATURE_NAMES) == NUM_FEATURES
