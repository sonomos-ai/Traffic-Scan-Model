# Changelog

<!-- Copyright © 2026 Sonomos, Inc. All rights reserved. -->

## [1.2] - 2026-04-14

### Added
- **huginn-net-tls integration** (Rust): Replaced manual ClientHello parsing and
  JA4 decomposition with `huginn-net-tls` v1.5 crate. The daemon now extracts
  TLS metadata, JA4 fingerprints, extension flags, and SNI from raw ClientHello
  bytes via a validated, type-safe parser (MIT/Apache-2.0, pure Rust, no C deps).
  - New `extract_tls_metadata()` function bridges huginn-net-tls output to our
    `TlsMetadata` and `Ja4Components` structs
  - New `build_feature_vector()` computes the full 61-dim vector from flow stats
    + huginn-net-tls output in a single call
  - New `classify_flow()` high-level API: raw flow data + ClientHello bytes in,
    classification result out
  - Complete Rust-side feature encoding (flow stats, TLS, JA4, SNI n-gram hash)
    with MurmurHash3 implementation verified against Python for cross-language
    determinism
- **cicflowmeter integration** (Python): New `scripts/extract_with_cicflowmeter.py`
  replaces manual pcap→CSV feature extraction with the `cicflowmeter` pip package
  (v0.5+, MIT). Maps CICFlowMeter's 77 bidirectional flow features to our 61-dim
  vector format, including upstream/downstream stats and byte/packet ratios.
  - Supports single pcap files and batch directory processing
  - Supports per-file labels via JSON label file
  - Supplemental TLS metadata log support for populating features [32:50]
  - Handles CICFlowMeter column name variations across versions

### Changed
- `classifier.rs` rewritten to integrate huginn-net-tls and include full
  Rust-side feature encoding. Previous version required features to be
  pre-computed externally; new version can extract them from raw packet data.
- Cargo.toml dependency: `huginn-net-tls = "1.5"` added alongside `tract-onnx`

## [1.1] - 2026-04-14

### Changed
- **Feature vector expanded from 40 → 61 dimensions** for improved classification accuracy
- Model architecture scaled from 40→64→32→1 (~5K params) to 61→96→48→1 (~11K params)
- SNI n-gram hash expanded from 3 → 11 dimensions to reduce hash collisions between
  distinct AI provider domains (e.g., api.openai.com vs generativelanguage.googleapis.com)

### Added
- **Directional flow statistics** (8 new features): upstream/downstream packet size
  mean/std/p50, upstream-to-downstream byte ratio, upstream-to-downstream packet count
  ratio. AI traffic has a distinctive asymmetry (small prompt upstream, many small
  streaming chunks downstream) that these features capture directly.
- **TLS extension fingerprint flags** (5 new features): has_sni_extension,
  has_sct_extension (signed_certificate_timestamp), has_status_request (OCSP stapling),
  has_supported_versions_13_only, has_post_handshake_auth. Programmatic API clients
  exhibit different extension profiles than browsers.
- New `_directional_stats()` helper for computing per-direction packet statistics
- New `FlowStats` fields: `upstream_packet_sizes`, `downstream_packet_sizes`,
  `upstream_bytes`, `downstream_bytes`
- New `TLSMetadata` fields: `has_sni_extension`, `has_sct_extension`,
  `has_status_request`, `has_supported_versions_13_only`, `has_post_handshake_auth`
- Tests for all new features, directional stats, TLS extension flags, expanded SNI hash

### Breaking Changes
- Feature vector dimension changed from 40 → 61 (all downstream consumers must update)
- ONNX model input shape changed from (1, 40) to (1, 61)
- Rust `NUM_FEATURES` constant updated from 40 to 61
- Synthetic training data CSV format changed (new columns)
- Previously trained models are incompatible; retrain required

## [1.0] - 2026-04-14

### Added
- Initial model definition: 40→64→32→1 MLP with BatchNorm and Dropout (4,929 params)
- Focal loss implementation for class-imbalanced binary classification
- Knowledge distillation loss (XGBoost teacher → MLP student)
- Feature extraction module with 40-dim vector:
  - Flow statistics (16 dims): packet size stats, IAT stats, duration, counts, throughput
  - First-N packet sizes (8 dims): first 8 packets for handshake fingerprinting
  - TLS metadata (7 dims): version, ciphers, extensions, ALPN, gRPC, H2, cert chain
  - JA4 components (6 dims): decomposed fingerprint with hashed cipher/ext lists
  - SNI n-gram hash (3 dims): character 2/3-gram hashing with MurmurHash3
- Synthetic data generator for pipeline testing
- Training script with focal loss, cosine annealing, stratified k-fold CV, early stopping
- XGBoost teacher training script for knowledge distillation
- ONNX validation script (opset, shapes, operators, latency benchmark)
- Rust/tract integration module with domain-level result cache
- Unit tests for model, loss functions, and all feature extractors
- README with full usage documentation
