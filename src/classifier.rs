// Copyright © 2026 Sonomos, Inc.
// All rights reserved.

//! Sonomos Traffic Classifier — Rust/tract inference integration
//!
//! This module provides the Stage 3 ML classifier for the traffic scanning
//! pipeline. Load once at daemon startup, call `classify()` per-flow when
//! Stages 1-2 are inconclusive.
//!
//! # Dependencies (Cargo.toml)
//! ```toml
//! [dependencies]
//! tract-onnx = "0.22"
//! anyhow = "1"
//! ```

use anyhow::Result;
use std::collections::HashMap;
use std::sync::RwLock;
use tract_onnx::prelude::*;

/// Number of input features expected by the ONNX model.
const NUM_FEATURES: usize = 40;

/// Default sensitivity threshold. Probability above this → AI traffic.
const DEFAULT_THRESHOLD: f32 = 0.5;

/// Per-domain result cache. Keyed by SNI domain.
struct DomainCache {
    cache: RwLock<HashMap<String, f32>>,
}

impl DomainCache {
    fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::with_capacity(1024)),
        }
    }

    fn get(&self, domain: &str) -> Option<f32> {
        self.cache.read().ok()?.get(domain).copied()
    }

    fn set(&self, domain: String, probability: f32) {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(domain, probability);
        }
    }
}

/// The Stage 3 ML classifier.
///
/// Wraps a tract-optimized ONNX model with a per-domain result cache.
/// Thread-safe: the optimized model plan is immutable after construction.
pub struct TrafficClassifier {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    cache: DomainCache,
    threshold: f32,
}

impl TrafficClassifier {
    /// Load the ONNX model from disk. Call once at daemon startup.
    ///
    /// The model is optimized during loading (BatchNorm folding, constant
    /// propagation, operator fusion). This takes ~10-50ms but only happens once.
    pub fn load(model_path: &str, threshold: Option<f32>) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, NUM_FEATURES)),
            )?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            model,
            cache: DomainCache::new(),
            threshold: threshold.unwrap_or(DEFAULT_THRESHOLD),
        })
    }

    /// Classify a flow's feature vector.
    ///
    /// Returns `(probability, is_ai_traffic)`.
    /// Checks the domain cache first; on miss, runs inference and caches.
    ///
    /// # Arguments
    /// * `features` - 40-dimension feature vector (must be pre-normalized)
    /// * `domain` - SNI domain for caching (empty string to skip cache)
    pub fn classify(&self, features: &[f32; NUM_FEATURES], domain: &str) -> Result<(f32, bool)> {
        // Check cache first
        if !domain.is_empty() {
            if let Some(cached_prob) = self.cache.get(domain) {
                return Ok((cached_prob, cached_prob > self.threshold));
            }
        }

        // Run inference
        let input = tract_ndarray::Array2::from_shape_vec(
            (1, NUM_FEATURES),
            features.to_vec(),
        )?;

        let output = self.model.run(tvec![input.into_tensor()])?;
        let logit = output[0].to_array_view::<f32>()?[[0, 0]];

        // Apply sigmoid: P(AI) = 1 / (1 + exp(-logit))
        let probability = 1.0 / (1.0 + (-logit).exp());
        let is_ai = probability > self.threshold;

        // Cache result
        if !domain.is_empty() {
            self.cache.set(domain.to_string(), probability);
        }

        Ok((probability, is_ai))
    }

    /// Update the sensitivity threshold at runtime (e.g., from user settings).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Clear the domain cache (e.g., when the model is retrained).
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.cache.write() {
            cache.clear();
        }
    }

    /// Return the number of cached domain results.
    pub fn cache_size(&self) -> usize {
        self.cache.cache.read().map(|c| c.len()).unwrap_or(0)
    }
}

// Example integration with the three-stage pipeline:
//
// ```rust
// fn classify_flow(flow: &Flow, classifier: &TrafficClassifier) -> TrafficDecision {
//     // Stage 1: Deterministic rules
//     if let Some(decision) = check_allowlist(&flow.domain) {
//         return decision;
//     }
//
//     // Stage 2: Heuristic scoring
//     let heuristic_score = compute_heuristic_score(flow);
//     if heuristic_score > HIGH_CONFIDENCE_THRESHOLD {
//         return TrafficDecision::Ai(heuristic_score);
//     }
//     if heuristic_score < LOW_CONFIDENCE_THRESHOLD {
//         return TrafficDecision::Normal(heuristic_score);
//     }
//
//     // Stage 3: ML classifier (only for inconclusive flows)
//     let features = extract_features(flow); // Your 40-dim vector
//     let (prob, is_ai) = classifier.classify(&features, &flow.domain)
//         .unwrap_or((0.0, false));
//
//     if is_ai {
//         TrafficDecision::Ai(prob)
//     } else {
//         TrafficDecision::Normal(prob)
//     }
// }
// ```

#[cfg(test)]
mod tests {
    use super::*;

    // These tests require the ONNX model file to exist.
    // Run `python scripts/train.py` first, then `cargo test`.

    #[test]
    fn test_classifier_loads() {
        let result = TrafficClassifier::load("models/traffic_classifier.onnx", None);
        assert!(result.is_ok(), "Failed to load model: {:?}", result.err());
    }

    #[test]
    fn test_inference_produces_valid_probability() {
        let classifier = TrafficClassifier::load("models/traffic_classifier.onnx", None)
            .expect("Failed to load model");

        let features = [0.0f32; NUM_FEATURES];
        let (prob, _is_ai) = classifier.classify(&features, "test.example.com")
            .expect("Inference failed");

        assert!(prob >= 0.0 && prob <= 1.0, "Probability out of range: {}", prob);
    }

    #[test]
    fn test_cache_works() {
        let classifier = TrafficClassifier::load("models/traffic_classifier.onnx", None)
            .expect("Failed to load model");

        let features = [0.5f32; NUM_FEATURES];
        let domain = "cached.example.com";

        // First call: cache miss → inference
        let (prob1, _) = classifier.classify(&features, domain).unwrap();
        assert_eq!(classifier.cache_size(), 1);

        // Second call: cache hit → same result
        let (prob2, _) = classifier.classify(&features, domain).unwrap();
        assert_eq!(prob1, prob2);
    }

    #[test]
    fn test_threshold_update() {
        let mut classifier = TrafficClassifier::load("models/traffic_classifier.onnx", None)
            .expect("Failed to load model");

        let features = [0.0f32; NUM_FEATURES];

        classifier.set_threshold(0.01); // very sensitive
        let (_, is_ai_sensitive) = classifier.classify(&features, "").unwrap();

        classifier.set_threshold(0.99); // very conservative
        let (_, is_ai_conservative) = classifier.classify(&features, "").unwrap();

        // With a threshold near 0, more things are classified as AI
        // With a threshold near 1, fewer things are
        // (exact results depend on the trained model)
        assert!(
            !(is_ai_sensitive == false && is_ai_conservative == true),
            "Lower threshold should not produce fewer AI detections"
        );
    }
}
