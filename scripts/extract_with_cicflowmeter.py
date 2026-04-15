# Copyright © 2026 Sonomos, Inc.
# All rights reserved.

"""
Extract training data from pcap files using cicflowmeter.

cicflowmeter computes 77 bidirectional flow features from pcaps — including
the upstream/downstream packet size stats, IATs, and byte counts that our
61-dim feature vector needs. This script maps CICFlowMeter output columns
to our feature format and appends the TLS/JA4/SNI features that cicflowmeter
does not provide (those must come from a separate TLS log or be left as
defaults for non-TLS flows).

Usage:
    pip install cicflowmeter

    # Extract from a single pcap (label all flows as AI=1 or normal=0)
    python scripts/extract_with_cicflowmeter.py \\
        --pcap data/captures/openai_traffic.pcap \\
        --label 1 \\
        --output data/openai_flows.csv

    # Extract from a directory of labeled pcaps
    python scripts/extract_with_cicflowmeter.py \\
        --pcap-dir data/captures/ \\
        --label-file data/captures/labels.json \\
        --output data/real_train.csv

    # labels.json format:
    # { "openai_traffic.pcap": 1, "youtube_browsing.pcap": 0, ... }

Requirements:
    pip install cicflowmeter>=0.5.0 --break-system-packages
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import FEATURE_NAMES, NUM_FEATURES, sni_ngram_hash, _HASH_DIMS


# Mapping from CICFlowMeter column names to our feature vector positions.
# CICFlowMeter produces ~77 columns; we use a subset and compute the rest.
#
# Our layout:
#   [0:16]  Flow stats (pkt sizes, IATs, duration, counts, throughput)
#   [16:24] Directional stats (up/down pkt size stats, byte/pkt ratios)
#   [24:32] First-N packet sizes (not available from CICFlowMeter)
#   [32:44] TLS metadata (not available from CICFlowMeter)
#   [44:50] JA4 components (not available from CICFlowMeter)
#   [50:61] SNI n-gram hash (derived from dst_ip or supplemental TLS log)

# CICFlowMeter column name → (our feature index, normalization divisor)
CICFLOW_TO_FEATURE = {
    # Packet size stats [0:7] — CICFlowMeter uses "Pkt Size" or "Pkt Len"
    "Pkt Size Avg":   (0, 1500.0),
    "Pkt Size Std":   (1, 1500.0),
    "Pkt Size Min":   (2, 1500.0),
    "Pkt Size Max":   (3, 1500.0),

    # IAT stats [7:12]
    "Flow IAT Mean":  (7, 1e7),   # CICFlowMeter reports IAT in microseconds
    "Flow IAT Std":   (8, 1e7),
    "Flow IAT Min":   (9, 1e7),
    "Flow IAT Max":   (10, 1e7),

    # Duration [12]
    "Flow Duration":  (12, None),  # special: log-scaled

    # Throughput [15]
    "Flow Byts/s":    (15, None),  # special: log-scaled

    # Directional upstream (forward in CICFlowMeter) [16:19]
    "Fwd Pkt Len Mean": (16, 1500.0),
    "Fwd Pkt Len Std":  (17, 1500.0),

    # Directional downstream (backward in CICFlowMeter) [19:22]
    "Bwd Pkt Len Mean": (19, 1500.0),
    "Bwd Pkt Len Std":  (20, 1500.0),
}

# Additional columns used for derived features
DERIVED_COLUMNS = {
    "Tot Fwd Pkts": "fwd_pkt_count",
    "Tot Bwd Pkts": "bwd_pkt_count",
    "TotLen Fwd Pkts": "fwd_bytes",
    "TotLen Bwd Pkts": "bwd_bytes",
    "Fwd Pkt Len Max": "fwd_pkt_max",
    "Bwd Pkt Len Max": "bwd_pkt_max",
    "Dst IP": "dst_ip",
    "Dst Port": "dst_port",
    "Src IP": "src_ip",
    "Flow ID": "flow_id",
}


def run_cicflowmeter(pcap_path: str, output_dir: str) -> str:
    """
    Run cicflowmeter on a pcap file and return the path to the output CSV.
    """
    pcap_name = Path(pcap_path).stem
    output_csv = Path(output_dir) / f"{pcap_name}_flows.csv"

    cmd = [
        "cicflowmeter",
        "-f", str(pcap_path),
        "-c", str(output_csv),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  WARNING: cicflowmeter returned {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")

    if not output_csv.exists():
        raise FileNotFoundError(f"cicflowmeter did not produce {output_csv}")

    return str(output_csv)


def normalize_column_names(header: list[str]) -> dict[str, int]:
    """
    CICFlowMeter column names vary between versions. Build a flexible
    lookup by normalizing to lowercase with spaces stripped.
    """
    lookup = {}
    for i, col in enumerate(header):
        normalized = col.strip().lower().replace("_", " ").replace("-", " ")
        lookup[normalized] = i
        # Also store original
        lookup[col.strip()] = i
    return lookup


def find_column(lookup: dict[str, int], *candidates: str) -> int | None:
    """Find a column index by trying multiple candidate names."""
    for name in candidates:
        if name in lookup:
            return lookup[name]
        normalized = name.strip().lower().replace("_", " ").replace("-", " ")
        if normalized in lookup:
            return lookup[normalized]
    return None


def safe_float(value: str, default: float = 0.0) -> float:
    """Parse a float, returning default on any error."""
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


def cicflow_row_to_features(
    row: list[str],
    col_lookup: dict[str, int],
    sni_domain: str = "",
) -> np.ndarray:
    """
    Convert a single CICFlowMeter CSV row to our 61-dim feature vector.

    Features that CICFlowMeter cannot provide (TLS metadata, JA4, first-N
    packet sizes) are left at 0.0 and must be populated from a supplemental
    TLS metadata log if available.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    def get_val(col_name: str) -> float:
        idx = find_column(col_lookup, col_name)
        if idx is not None and idx < len(row):
            return safe_float(row[idx])
        return 0.0

    # --- Flow statistics [0:16] ---

    # Packet size stats
    features[0] = get_val("Pkt Size Avg") / 1500.0
    features[1] = get_val("Pkt Size Std") / 1500.0
    features[2] = get_val("Pkt Size Min") / 1500.0
    features[3] = get_val("Pkt Size Max") / 1500.0
    # p25/p50/p75 not directly available; approximate from mean±std
    mean_pkt = get_val("Pkt Size Avg")
    std_pkt = get_val("Pkt Size Std")
    features[4] = max(mean_pkt - 0.675 * std_pkt, 0) / 1500.0  # p25 approx
    features[5] = mean_pkt / 1500.0  # p50 ≈ mean for symmetric
    features[6] = (mean_pkt + 0.675 * std_pkt) / 1500.0  # p75 approx

    # IAT stats (CICFlowMeter reports in microseconds, normalize to [0,1])
    features[7] = get_val("Flow IAT Mean") / 1e7
    features[8] = get_val("Flow IAT Std") / 1e7
    features[9] = get_val("Flow IAT Min") / 1e7
    features[10] = get_val("Flow IAT Max") / 1e7
    # p50 ≈ mean
    features[11] = features[7]

    # Duration (log-scaled; CICFlowMeter reports in microseconds)
    duration_us = get_val("Flow Duration")
    duration_s = duration_us / 1e6
    features[12] = np.log1p(min(duration_s, 300.0)) / np.log1p(300.0)

    # Packet counts (log-scaled)
    fwd_pkts = get_val("Tot Fwd Pkts")
    bwd_pkts = get_val("Tot Bwd Pkts")
    features[13] = np.log1p(fwd_pkts) / np.log1p(10000)
    features[14] = np.log1p(bwd_pkts) / np.log1p(10000)

    # Throughput (log-scaled)
    bps = get_val("Flow Byts/s")
    features[15] = np.log1p(bps) / np.log1p(1e9)

    # --- Directional stats [16:24] ---

    features[16] = get_val("Fwd Pkt Len Mean") / 1500.0
    features[17] = get_val("Fwd Pkt Len Std") / 1500.0
    features[18] = get_val("Fwd Pkt Len Mean") / 1500.0  # p50 ≈ mean

    features[19] = get_val("Bwd Pkt Len Mean") / 1500.0
    features[20] = get_val("Bwd Pkt Len Std") / 1500.0
    features[21] = get_val("Bwd Pkt Len Mean") / 1500.0  # p50 ≈ mean

    # Byte ratio
    fwd_bytes = get_val("TotLen Fwd Pkts")
    bwd_bytes = get_val("TotLen Bwd Pkts")
    total_bytes = fwd_bytes + bwd_bytes
    features[22] = fwd_bytes / total_bytes if total_bytes > 0 else 0.5

    # Packet count ratio
    total_pkts = fwd_pkts + bwd_pkts
    features[23] = fwd_pkts / total_pkts if total_pkts > 0 else 0.5

    # --- First-N packet sizes [24:32] — not available from CICFlowMeter ---
    # Leave as 0.0; will be populated from supplemental data if available

    # --- TLS metadata [32:44] — not available from CICFlowMeter ---
    # Leave as 0.0; populate from supplemental TLS log

    # --- JA4 components [44:50] — not available from CICFlowMeter ---
    # Leave as 0.0; populate from supplemental TLS log

    # --- SNI n-gram hash [50:61] ---
    if sni_domain:
        sni_hash = sni_ngram_hash(sni_domain, dims=_HASH_DIMS)
        features[50:50 + _HASH_DIMS] = sni_hash

    return features


def process_pcap(
    pcap_path: str,
    label: int,
    tls_log: dict | None = None,
    sni_override: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a pcap file through CICFlowMeter and extract labeled features.

    Args:
        pcap_path: Path to pcap file
        label: 1 for AI traffic, 0 for normal
        tls_log: Optional dict mapping flow_id → {sni, tls_version, ...}
        sni_override: If set, use this domain for all flows' SNI hash

    Returns:
        X: Feature matrix (n_flows, 61)
        y: Labels (n_flows,)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cic_csv = run_cicflowmeter(pcap_path, tmpdir)

        with open(cic_csv) as f:
            reader = csv.reader(f)
            header = next(reader)
            col_lookup = normalize_column_names(header)
            rows = list(reader)

    print(f"  Extracted {len(rows)} flows from {pcap_path}")

    X_list = []
    for row in rows:
        # Determine SNI for this flow
        sni = sni_override
        if tls_log and not sni:
            flow_id_idx = find_column(col_lookup, "Flow ID")
            if flow_id_idx is not None and flow_id_idx < len(row):
                flow_id = row[flow_id_idx].strip()
                sni = tls_log.get(flow_id, {}).get("sni", "")

        features = cicflow_row_to_features(row, col_lookup, sni)
        X_list.append(features)

    if not X_list:
        return np.empty((0, NUM_FEATURES)), np.empty(0)

    X = np.stack(X_list)
    y = np.full(len(X), label, dtype=np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Extract training features from pcap files using cicflowmeter"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcap", type=str, help="Single pcap file path")
    group.add_argument("--pcap-dir", type=str, help="Directory of pcap files")

    parser.add_argument("--label", type=int, choices=[0, 1],
                        help="Label for single pcap (0=normal, 1=AI)")
    parser.add_argument("--label-file", type=str,
                        help="JSON file mapping pcap filenames to labels")
    parser.add_argument("--sni", type=str, default="",
                        help="Override SNI domain for all flows")
    parser.add_argument("--tls-log", type=str, default=None,
                        help="JSON file mapping flow_id → TLS metadata")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    # Load supplemental data
    tls_log = None
    if args.tls_log:
        with open(args.tls_log) as f:
            tls_log = json.load(f)

    all_X = []
    all_y = []

    if args.pcap:
        if args.label is None:
            parser.error("--label is required when using --pcap")
        print(f"Processing {args.pcap} (label={args.label})...")
        X, y = process_pcap(args.pcap, args.label, tls_log, args.sni)
        all_X.append(X)
        all_y.append(y)

    elif args.pcap_dir:
        if args.label_file is None and args.label is None:
            parser.error("--label or --label-file is required when using --pcap-dir")

        labels = {}
        if args.label_file:
            with open(args.label_file) as f:
                labels = json.load(f)

        pcap_dir = Path(args.pcap_dir)
        pcap_files = sorted(
            list(pcap_dir.glob("*.pcap")) + list(pcap_dir.glob("*.pcapng"))
        )

        if not pcap_files:
            print(f"No pcap files found in {pcap_dir}")
            sys.exit(1)

        for pcap_file in pcap_files:
            file_label = labels.get(pcap_file.name, args.label)
            if file_label is None:
                print(f"  Skipping {pcap_file.name} (no label)")
                continue

            print(f"Processing {pcap_file.name} (label={file_label})...")
            X, y = process_pcap(str(pcap_file), file_label, tls_log, args.sni)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

    if not all_X:
        print("No flows extracted.")
        sys.exit(1)

    X_full = np.vstack(all_X)
    y_full = np.concatenate(all_y)

    # Write output CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["label"])
        for i in range(len(X_full)):
            writer.writerow([f"{v:.6f}" for v in X_full[i]] + [f"{y_full[i]:.0f}"])

    n_ai = int(y_full.sum())
    n_normal = len(y_full) - n_ai
    print(f"\nSaved to {output_path}")
    print(f"  Total: {len(y_full)}, AI: {n_ai}, Normal: {n_normal}")
    print(f"  Features: {X_full.shape[1]}")
    print(f"\nNote: TLS metadata features [32:50] and first-N packet sizes [24:32]")
    print(f"are zeroed unless a --tls-log is provided. For best accuracy,")
    print(f"capture TLS metadata separately using the Sonomos daemon's logging")
    print(f"mode and provide it via --tls-log.")


if __name__ == "__main__":
    main()
