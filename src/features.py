# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Feature extraction for the Sonomos Traffic Classifier.

Converts raw flow metadata into a 61-dimension numeric feature vector:
  [0:24]  Flow statistics (packet sizes, IATs, duration, counts, throughput,
          directional stats, asymmetry ratios)
  [24:32] First-N packet sizes (first 8 packets, upstream/downstream interleaved)
  [32:44] TLS handshake metadata (7 base + 5 extension fingerprint flags)
  [44:50] JA4 fingerprint components
  [50:61] SNI character n-gram hash (11 dimensions)
"""

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


NUM_FEATURES = 61

# --- SNI n-gram hashing ---

_HASH_DIMS = 11  # expanded from 3 → 11 for better domain discrimination


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

    Expanded from 3 → 11 dims to reduce collisions. AI provider domains
    (api.openai.com, generativelanguage.googleapis.com, api.anthropic.com)
    have very different n-gram distributions that were crushed together at 3 dims.

    Args:
        domain: SNI hostname (e.g., "api.openai.com")
        dims: Output vector dimensionality (default 11)

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
    # TLS extension fingerprint flags
    has_sni_extension: bool = True
    has_sct_extension: bool = False           # signed_certificate_timestamp
    has_status_request: bool = False          # OCSP stapling
    has_supported_versions_13_only: bool = False  # only lists TLS 1.3
    has_post_handshake_auth: bool = False     # common in programmatic API clients


def tls_to_features(tls: TLSMetadata) -> np.ndarray:
    """
    Convert TLS metadata to 12-dim numeric vector.

    [0]  TLS version ordinal (normalized)
    [1]  Cipher suite count (normalized)
    [2]  Extension count (normalized)
    [3]  ALPN ordinal (normalized)
    [4]  Has gRPC ALPN (binary)
    [5]  Has H2 ALPN (binary)
    [6]  Certificate chain length (normalized)
    [7]  Has SNI extension (binary)
    [8]  Has SCT extension (binary)
    [9]  Has OCSP status_request (binary)
    [10] Has supported_versions with TLS 1.3 only (binary)
    [11] Has post_handshake_auth (binary)
    """
    features = np.zeros(12, dtype=np.float32)
    features[0] = _TLS_VERSION_ORD.get(tls.version, 3) / 4.0
    features[1] = min(tls.cipher_suite_count / 30.0, 1.0)
    features[2] = min(tls.extension_count / 30.0, 1.0)
    features[3] = _ALPN_ORD.get(tls.alpn, 0) / 5.0
    features[4] = 1.0 if tls.has_grpc_alpn else 0.0
    features[5] = 1.0 if tls.has_h2_alpn else 0.0
    features[6] = min(tls.cert_chain_length / 5.0, 1.0)
    # Extension fingerprint flags
    features[7] = 1.0 if tls.has_sni_extension else 0.0
    features[8] = 1.0 if tls.has_sct_extension else 0.0
    features[9] = 1.0 if tls.has_status_request else 0.0
    features[10] = 1.0 if tls.has_supported_versions_13_only else 0.0
    features[11] = 1.0 if tls.has_post_handshake_auth else 0.0
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
    # Directional packet sizes for asymmetry analysis
    upstream_packet_sizes: list[int] = field(default_factory=list)
    downstream_packet_sizes: list[int] = field(default_factory=list)
    upstream_bytes: int = 0
    downstream_bytes: int = 0


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


def _directional_stats(sizes: list[int]) -> tuple[float, float, float]:
    """Compute mean, std, p50 for a directional packet size list. Returns (0,0,0) if empty."""
    if not sizes:
        return (0.0, 0.0, 0.0)
    arr = np.array(sizes, dtype=np.float32)
    return (
        float(np.mean(arr)) / 1500.0,
        float(np.std(arr)) / 1500.0,
        float(np.median(arr)) / 1500.0,
    )


def flow_to_features(flow: FlowStats) -> np.ndarray:
    """
    Convert flow statistics to 24-dim numeric vector.

    [0:7]   Packet size stats (mean/std/min/max/p25/p50/p75), normalized by /1500
    [7:12]  IAT stats (mean/std/min/max/p50), normalized by /10.0 (seconds)
    [12]    Flow duration (log-scaled)
    [13]    Upstream packet count (log-scaled)
    [14]    Downstream packet count (log-scaled)
    [15]    Bytes per second (log-scaled)
    [16:19] Upstream packet size stats (mean/std/p50), normalized by /1500
    [19:22] Downstream packet size stats (mean/std/p50), normalized by /1500
    [22]    Upstream-to-downstream byte ratio
    [23]    Upstream-to-downstream packet count ratio
    """
    features = np.zeros(24, dtype=np.float32)

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

    # --- Directional features [16:24] ---

    # Upstream packet size stats (mean, std, p50)
    up_mean, up_std, up_p50 = _directional_stats(flow.upstream_packet_sizes)
    features[16] = up_mean
    features[17] = up_std
    features[18] = up_p50

    # Downstream packet size stats (mean, std, p50)
    down_mean, down_std, down_p50 = _directional_stats(flow.downstream_packet_sizes)
    features[19] = down_mean
    features[20] = down_std
    features[21] = down_p50

    # Byte ratio: upstream / (upstream + downstream). AI traffic is asymmetric
    # (small prompt upstream, large streaming response downstream → low ratio)
    total_directional = flow.upstream_bytes + flow.downstream_bytes
    if total_directional > 0:
        features[22] = flow.upstream_bytes / total_directional
    else:
        features[22] = 0.5  # neutral default

    # Packet count ratio: upstream / (upstream + downstream)
    total_pkts = flow.packet_count_upstream + flow.packet_count_downstream
    if total_pkts > 0:
        features[23] = flow.packet_count_upstream / total_pkts
    else:
        features[23] = 0.5

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
    Extract the full 61-dim feature vector from a flow record.

    Returns np.ndarray of shape (61,) with all features normalized to ~[0, 1].
    """
    flow_feats = flow_to_features(record.flow_stats)           # 24 dims
    first_n_feats = first_n_to_features(
        record.flow_stats.first_n_packet_sizes, n=8
    )                                                           # 8 dims
    tls_feats = tls_to_features(record.tls_metadata)            # 12 dims
    ja4_feats = ja4_to_features(record.ja4_components)          # 6 dims
    sni_feats = sni_ngram_hash(record.sni_domain, dims=_HASH_DIMS)  # 11 dims

    combined = np.concatenate([
        flow_feats,     # [0:24]
        first_n_feats,  # [24:32]
        tls_feats,      # [32:44]
        ja4_feats,      # [44:50]
        sni_feats,      # [50:61]
    ])

    assert combined.shape == (NUM_FEATURES,), f"Expected {NUM_FEATURES} features, got {combined.shape[0]}"
    return combined


# Feature names for debugging and analysis
FEATURE_NAMES = [
    # Flow statistics [0:24]
    "pkt_size_mean", "pkt_size_std", "pkt_size_min", "pkt_size_max",
    "pkt_size_p25", "pkt_size_p50", "pkt_size_p75",
    "iat_mean", "iat_std", "iat_min", "iat_max", "iat_p50",
    "duration", "pkt_count_up", "pkt_count_down", "bytes_per_sec",
    "up_pkt_size_mean", "up_pkt_size_std", "up_pkt_size_p50",
    "down_pkt_size_mean", "down_pkt_size_std", "down_pkt_size_p50",
    "byte_ratio_up", "pkt_ratio_up",
    # First-N packet sizes [24:32]
    "first_pkt_1", "first_pkt_2", "first_pkt_3", "first_pkt_4",
    "first_pkt_5", "first_pkt_6", "first_pkt_7", "first_pkt_8",
    # TLS metadata [32:44]
    "tls_version", "tls_cipher_count", "tls_ext_count", "tls_alpn",
    "tls_has_grpc", "tls_has_h2", "tls_cert_chain_len",
    "tls_has_sni", "tls_has_sct", "tls_has_status_req",
    "tls_13_only", "tls_post_handshake_auth",
    # JA4 components [44:50]
    "ja4_tls_ver", "ja4_cipher_count", "ja4_ext_count", "ja4_alpn",
    "ja4_cipher_hash_0", "ja4_cipher_hash_1",
    # SNI n-gram hash [50:61]
    "sni_hash_0", "sni_hash_1", "sni_hash_2", "sni_hash_3",
    "sni_hash_4", "sni_hash_5", "sni_hash_6", "sni_hash_7",
    "sni_hash_8", "sni_hash_9", "sni_hash_10",
]

assert len(FEATURE_NAMES) == NUM_FEATURES
