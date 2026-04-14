# Changelog

<!-- Copyright © 2026 Sonomos, Inc. All rights reserved. -->

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
