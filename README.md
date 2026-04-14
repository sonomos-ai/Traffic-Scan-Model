# Sonomos Traffic Classifier

<!-- Copyright © 2026 Sonomos, Inc. All rights reserved. -->

A tiny MLP binary classifier (4,737 parameters) for detecting AI provider network
traffic from TLS/HTTPS metadata. Designed for sub-100μs inference via
[tract](https://github.com/sonos/tract) (pure Rust, no C++ deps).

## Architecture

```
Input (40 features) → Linear(64) → BatchNorm → ReLU → Dropout(0.1)
                     → Linear(32) → BatchNorm → ReLU → Dropout(0.1)
                     → Linear(1)  → Sigmoid → P(AI traffic)
```

**Stage 3** of the Sonomos Desktop three-stage traffic scanning pipeline:

1. **Stage 1** — Deterministic rules (sub-μs): domain allowlist + user overrides + cache
2. **Stage 2** — Heuristic scoring (~μs): JA4 fingerprint, SNI pattern, DNS/IP correlation
3. **Stage 3** — This ML classifier (~10–70ms cold, <100μs warm): ONNX model via tract

## Feature Vector (40 dimensions)

| Group | Features | Dims |
|---|---|---|
| Flow statistics | pkt size mean/std/min/max/p25/p50/p75, IAT mean/std/min/max/p50, duration, pkt count (up/down), bytes/sec | 16 |
| First-N packet sizes | first 8 packet sizes (upstream interleaved with downstream) | 8 |
| TLS metadata | version, cipher count, ext count, ALPN, has_grpc, has_h2, cert_chain_len | 7 |
| JA4 components | version_ord, cipher_count, ext_count, alpn_ord, sorted_cipher_hash(2d), sorted_ext_hash(1d) | 6 |
| SNI n-gram hash | character 2/3-gram hashing into 3-dim feature vector | 3 |

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

Replace synthetic data with real captures:

```bash
# Collect labeled packet captures (requires tcpdump/tshark)
python scripts/capture_traffic.py --interface eth0 --duration 3600 --output data/captures/

# Extract features from captures
python scripts/extract_features.py --input data/captures/ --output data/real_train.csv

# Train on real data
python scripts/train.py --data data/real_train.csv --output models/traffic_classifier.onnx
```

## Tract Integration (Rust)

```toml
# Cargo.toml
[dependencies]
tract-onnx = "0.22"
```

```rust
use tract_onnx::prelude::*;

let model = tract_onnx::onnx()
    .model_for_path("traffic_classifier.onnx")?
    .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 40)))?
    .into_optimized()?
    .into_runnable()?;

let features: Vec<f32> = extract_features(&flow); // your 40-dim vector
let input = tract_ndarray::Array2::from_shape_vec((1, 40), features)?.into();
let output = model.run(tvec![input])?;
let p_ai = output[0].to_array_view::<f32>()?[[0, 0]];

if p_ai > sensitivity_threshold {
    // AI traffic detected
}
```

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
- Model size: ~20KB (FP32 ONNX)

## License

Proprietary — Sonomos, Inc.
