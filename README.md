# Sonomos Traffic Classifier

<!-- Copyright © 2026 Sonomos, Inc. All rights reserved. -->

A tiny MLP binary classifier (~11K parameters) for detecting AI provider network
traffic from TLS/HTTPS metadata. Designed for sub-100μs inference via
[tract](https://github.com/sonos/tract) (pure Rust, no C++ deps).

## Architecture

```
Input (61 features) → Linear(96) → BatchNorm → ReLU → Dropout(0.1)
                     → Linear(48) → BatchNorm → ReLU → Dropout(0.1)
                     → Linear(1)  → Sigmoid → P(AI traffic)
```

**Stage 3** of the Sonomos Desktop three-stage traffic scanning pipeline:

1. **Stage 1** — Deterministic rules (sub-μs): domain allowlist + user overrides + cache
2. **Stage 2** — Heuristic scoring (~μs): JA4 fingerprint, SNI pattern, DNS/IP correlation
3. **Stage 3** — This ML classifier (~10–70ms cold, <100μs warm): ONNX model via tract

## Feature Vector (61 dimensions)

| Group | Features | Dims |
|---|---|---|
| Flow statistics | pkt size mean/std/min/max/p25/p50/p75, IAT mean/std/min/max/p50, duration, pkt count (up/down), bytes/sec | 16 |
| Directional stats | upstream pkt size mean/std/p50, downstream pkt size mean/std/p50, byte ratio (up/total), pkt count ratio (up/total) | 8 |
| First-N packet sizes | first 8 packet sizes (upstream interleaved with downstream) | 8 |
| TLS metadata | version, cipher count, ext count, ALPN, has_grpc, has_h2, cert_chain_len, has_sni, has_sct, has_status_request, tls_13_only, post_handshake_auth | 12 |
| JA4 components | version_ord, cipher_count, ext_count, alpn_ord, sorted_cipher_hash(2d) | 6 |
| SNI n-gram hash | character 2/3-gram hashing into 11-dim feature vector | 11 |

## Quick Start

```bash
# Install deps
pip install torch scikit-learn onnx onnxruntime numpy pandas

# Generate synthetic training data (for testing the pipeline)
python scripts/generate_synthetic_data.py --output data/synthetic_train.csv --samples 10000

# Train
python scripts/train.py --data data/synthetic_train.csv --output models/traffic_classifier.onnx

# Validate ONNX export
python scripts/validate_onnx.py --model models/traffic_classifier.onnx

# Run tests
python -m pytest tests/ -v
```

## Real Data Collection

Extract features from real packet captures using [cicflowmeter](https://pypi.org/project/cicflowmeter/):

```bash
pip install cicflowmeter>=0.5.0

# Single pcap with label (1=AI traffic, 0=normal)
python scripts/extract_with_cicflowmeter.py \
    --pcap data/captures/openai_traffic.pcap \
    --label 1 \
    --sni api.openai.com \
    --output data/openai_flows.csv

# Batch: directory of pcaps with per-file labels
python scripts/extract_with_cicflowmeter.py \
    --pcap-dir data/captures/ \
    --label-file data/captures/labels.json \
    --output data/real_train.csv

# Train on real data
python scripts/train.py --data data/real_train.csv --output models/traffic_classifier.onnx
```

## Rust Integration (tract + huginn-net-tls)

```toml
# Cargo.toml
[dependencies]
huginn-net-tls = "1.5"
tract-onnx = "0.22"
anyhow = "1"
```

```rust
use crate::classifier::{TrafficClassifier, FlowStats};

// Load once at daemon startup
let classifier = TrafficClassifier::load("traffic_classifier.onnx", Some(0.5))?;

// On each intercepted flow: pass flow stats + raw ClientHello bytes
let (probability, is_ai, sni) = classifier.classify_flow(&flow_stats, &client_hello_bytes)?;

if is_ai {
    // AI traffic detected — apply Cloak interception
}
```

The `classify_flow()` method handles the full pipeline internally:
1. Passes ClientHello bytes through `huginn-net-tls` for JA4/TLS extraction
2. Builds the 61-dim feature vector (flow stats + TLS + JA4 + SNI hash)
3. Runs tract ONNX inference
4. Returns `(probability, is_ai, sni_domain)`

## XGBoost Distillation (Optional)

For maximum accuracy, train an XGBoost teacher first:

```bash
python scripts/train_xgboost_teacher.py --data data/real_train.csv --output models/xgb_teacher.json
python scripts/train.py --data data/real_train.csv --teacher models/xgb_teacher.json --output models/traffic_classifier.onnx
```

## Model Metrics

Target metrics (on real data):
- AUC-PR > 0.95
- F1 > 0.92
- Precision@90%Recall > 0.90
- Inference: <100μs (tract, x86_64)
- Model size: ~45KB (FP32 ONNX)

## License

Proprietary — Sonomos, Inc.
