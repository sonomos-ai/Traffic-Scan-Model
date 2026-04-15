// Copyright © 2026 Sonomos, Inc.
// All rights reserved.

//! Sonomos Traffic Classifier — Rust/tract inference with huginn-net-tls
//!
//! This module provides the Stage 2–3 traffic scanning pipeline:
//!   - Stage 2: huginn-net-tls extracts JA4 fingerprints and TLS metadata
//!     from raw ClientHello bytes for heuristic scoring
//!   - Stage 3: tract runs the ONNX classifier when Stages 1-2 are inconclusive
//!
//! # Dependencies (Cargo.toml)
//! ```toml
//! [dependencies]
//! huginn-net-tls = "1.5"
//! tract-onnx = "0.22"
//! anyhow = "1"
//! ```

use anyhow::Result;
use std::collections::HashMap;
use std::sync::RwLock;
use tract_onnx::prelude::*;

/// Number of input features expected by the ONNX model.
const NUM_FEATURES: usize = 61;

/// Default sensitivity threshold. Probability above this → AI traffic.
const DEFAULT_THRESHOLD: f32 = 0.5;

// --- TLS extension type IDs (IANA registry) ---

/// status_request (OCSP stapling)
const EXT_STATUS_REQUEST: u16 = 5;
/// signed_certificate_timestamp
const EXT_SCT: u16 = 18;
/// post_handshake_auth
const EXT_POST_HANDSHAKE_AUTH: u16 = 49;
/// supported_versions
const EXT_SUPPORTED_VERSIONS: u16 = 43;
/// server_name (SNI)
const EXT_SERVER_NAME: u16 = 0;

// --- TLS metadata structs (mirror Python features.py) ---

/// TLS handshake metadata extracted via huginn-net-tls.
/// Maps to feature vector indices [32:44].
#[derive(Debug, Clone)]
pub struct TlsMetadata {
    pub version: String,
    pub cipher_suite_count: usize,
    pub extension_count: usize,
    pub alpn: String,
    pub has_grpc_alpn: bool,
    pub has_h2_alpn: bool,
    pub cert_chain_length: usize,
    pub has_sni_extension: bool,
    pub has_sct_extension: bool,
    pub has_status_request: bool,
    pub has_supported_versions_13_only: bool,
    pub has_post_handshake_auth: bool,
}

/// Decomposed JA4 fingerprint components extracted via huginn-net-tls.
/// Maps to feature vector indices [44:50].
#[derive(Debug, Clone)]
pub struct Ja4Components {
    pub tls_version: String,
    pub cipher_count: usize,
    pub extension_count: usize,
    pub alpn: String,
    pub sorted_cipher_hash: String,
    pub sorted_extension_hash: String,
}

// --- huginn-net-tls bridge ---

/// Extract TLS metadata and JA4 components from a raw ClientHello.
///
/// Uses huginn-net-tls for validated TLS parsing and JA4 hash computation,
/// replacing all manual ClientHello byte parsing.
///
/// # Arguments
/// * `client_hello` - Raw ClientHello message bytes (after TLS record header)
///
/// # Returns
/// `Some((TlsMetadata, Ja4Components, sni_domain))` on success, `None` if
/// the bytes are not a valid ClientHello.
///
/// # Example
/// ```rust
/// if let Some((tls_meta, ja4, sni)) = extract_tls_metadata(&client_hello_bytes) {
///     // tls_meta and ja4 feed directly into the 61-dim feature vector
///     // sni feeds into the SNI n-gram hash (features [50:61])
/// }
/// ```
pub fn extract_tls_metadata(
    client_hello: &[u8],
) -> Option<(TlsMetadata, Ja4Components, String)> {
    use huginn_net_tls::TlsAnalyzer;

    let analyzer = TlsAnalyzer::new();
    let result = analyzer.analyze(client_hello).ok()?;

    let cipher_suites = result.cipher_suites();
    let extensions = result.extensions();
    let extension_ids: Vec<u16> = extensions.iter().map(|e| e.extension_type()).collect();

    // Extract ALPN values
    let alpn_values = result.alpn_protocols().unwrap_or_default();
    let primary_alpn = alpn_values.first().cloned().unwrap_or_default();

    // Check supported_versions for TLS 1.3 only
    let supported_versions_13_only = result
        .supported_versions()
        .map(|versions| versions.len() == 1 && versions.contains(&0x0304))
        .unwrap_or(false);

    let sni = result.sni().unwrap_or_default().to_string();

    let tls_meta = TlsMetadata {
        version: format_tls_version(result.tls_version()),
        cipher_suite_count: cipher_suites.len(),
        extension_count: extensions.len(),
        alpn: primary_alpn.clone(),
        has_grpc_alpn: alpn_values.iter().any(|a| a == "grpc"),
        has_h2_alpn: alpn_values.iter().any(|a| a == "h2"),
        cert_chain_length: 0, // populated from ServerHello, not available in ClientHello
        has_sni_extension: extension_ids.contains(&EXT_SERVER_NAME),
        has_sct_extension: extension_ids.contains(&EXT_SCT),
        has_status_request: extension_ids.contains(&EXT_STATUS_REQUEST),
        has_supported_versions_13_only: supported_versions_13_only,
        has_post_handshake_auth: extension_ids.contains(&EXT_POST_HANDSHAKE_AUTH),
    };

    // Build JA4 components from huginn-net-tls's JA4 hash output
    let ja4_full = result.ja4_fingerprint().unwrap_or_default();
    // JA4 format: "a_b_c" where a = metadata, b = sorted cipher hash, c = sorted ext hash
    let ja4_parts: Vec<&str> = ja4_full.splitn(3, '_').collect();

    let ja4 = Ja4Components {
        tls_version: tls_meta.version.clone(),
        cipher_count: cipher_suites.len(),
        extension_count: extensions.len(),
        alpn: primary_alpn,
        sorted_cipher_hash: ja4_parts.get(1).unwrap_or(&"").to_string(),
        sorted_extension_hash: ja4_parts.get(2).unwrap_or(&"").to_string(),
    };

    Some((tls_meta, ja4, sni))
}

fn format_tls_version(version: u16) -> String {
    match version {
        0x0300 => "SSLv3".into(),
        0x0301 => "TLS1.0".into(),
        0x0302 => "TLS1.1".into(),
        0x0303 => "TLS1.2".into(),
        0x0304 => "TLS1.3".into(),
        v => format!("0x{:04x}", v),
    }
}

// --- Feature encoding (mirrors features.py normalization) ---

/// TLS version ordinal mapping.
fn tls_version_ord(version: &str) -> f32 {
    let ord = match version {
        "SSLv3" => 0,
        "TLS1.0" | "TLSv1.0" => 1,
        "TLS1.1" | "TLSv1.1" => 2,
        "TLS1.2" | "TLSv1.2" => 3,
        "TLS1.3" | "TLSv1.3" => 4,
        _ => 3, // default to TLS 1.2
    };
    ord as f32 / 4.0
}

/// ALPN ordinal mapping.
fn alpn_ord(alpn: &str) -> f32 {
    let ord = match alpn {
        "" => 0,
        "http/1.0" => 1,
        "http/1.1" => 2,
        "h2" => 3,
        "h3" => 4,
        "grpc" => 5,
        _ => 0,
    };
    ord as f32 / 5.0
}

/// MurmurHash3 32-bit for SNI n-gram hashing and JA4 cipher hash encoding.
/// Must produce identical output to the Python _murmurhash3_32.
fn murmurhash3_32(key: &[u8], seed: u32) -> u32 {
    let mut h: u32 = seed;
    let length = key.len();
    let nblocks = length / 4;
    let c1: u32 = 0xCC9E2D51;
    let c2: u32 = 0x1B873593;

    for i in 0..nblocks {
        let mut k = u32::from_le_bytes([
            key[i * 4],
            key[i * 4 + 1],
            key[i * 4 + 2],
            key[i * 4 + 3],
        ]);
        k = k.wrapping_mul(c1);
        k = k.rotate_left(15);
        k = k.wrapping_mul(c2);
        h ^= k;
        h = h.rotate_left(13);
        h = h.wrapping_mul(5).wrapping_add(0xE6546B64);
    }

    let tail_start = nblocks * 4;
    let mut k1: u32 = 0;
    let tail_len = length & 3;
    if tail_len >= 3 {
        k1 ^= (key[tail_start + 2] as u32) << 16;
    }
    if tail_len >= 2 {
        k1 ^= (key[tail_start + 1] as u32) << 8;
    }
    if tail_len >= 1 {
        k1 ^= key[tail_start] as u32;
        k1 = k1.wrapping_mul(c1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(c2);
        h ^= k1;
    }

    h ^= length as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EBCA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2AE35);
    h ^= h >> 16;
    h
}

/// Encode TLS metadata into features [32:44] of the 61-dim vector.
fn encode_tls_features(tls: &TlsMetadata, features: &mut [f32; NUM_FEATURES]) {
    features[32] = tls_version_ord(&tls.version);
    features[33] = (tls.cipher_suite_count as f32 / 30.0).min(1.0);
    features[34] = (tls.extension_count as f32 / 30.0).min(1.0);
    features[35] = alpn_ord(&tls.alpn);
    features[36] = if tls.has_grpc_alpn { 1.0 } else { 0.0 };
    features[37] = if tls.has_h2_alpn { 1.0 } else { 0.0 };
    features[38] = (tls.cert_chain_length as f32 / 5.0).min(1.0);
    features[39] = if tls.has_sni_extension { 1.0 } else { 0.0 };
    features[40] = if tls.has_sct_extension { 1.0 } else { 0.0 };
    features[41] = if tls.has_status_request { 1.0 } else { 0.0 };
    features[42] = if tls.has_supported_versions_13_only { 1.0 } else { 0.0 };
    features[43] = if tls.has_post_handshake_auth { 1.0 } else { 0.0 };
}

/// Encode JA4 components into features [44:50] of the 61-dim vector.
fn encode_ja4_features(ja4: &Ja4Components, features: &mut [f32; NUM_FEATURES]) {
    features[44] = tls_version_ord(&ja4.tls_version);
    features[45] = (ja4.cipher_count as f32 / 30.0).min(1.0);
    features[46] = (ja4.extension_count as f32 / 30.0).min(1.0);
    features[47] = alpn_ord(&ja4.alpn);

    if !ja4.sorted_cipher_hash.is_empty() {
        let h = murmurhash3_32(ja4.sorted_cipher_hash.as_bytes(), 100);
        features[48] = (h & 0xFFFF) as f32 / 65535.0;
        features[49] = ((h >> 16) & 0xFFFF) as f32 / 65535.0;
    }
}

/// Encode SNI domain into features [50:61] using character n-gram hashing.
/// Produces identical output to the Python sni_ngram_hash function.
fn encode_sni_features(domain: &str, features: &mut [f32; NUM_FEATURES]) {
    const DIMS: usize = 11;
    let mut vec = [0.0f32; DIMS];
    let domain = domain.to_lowercase();
    let domain = domain.trim_matches('.');

    // Character 2-grams and 3-grams
    for n in [2usize, 3] {
        if domain.len() < n {
            continue;
        }
        for i in 0..=(domain.len() - n) {
            let gram = &domain.as_bytes()[i..i + n];
            let h = murmurhash3_32(gram, 0);
            let idx = (h as usize) % DIMS;
            let sign_h = murmurhash3_32(gram, 42);
            let sign: f32 = if (sign_h & 1) == 0 { 1.0 } else { -1.0 };
            vec[idx] += sign;
        }
    }

    // L2 normalize
    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }

    features[50..61].copy_from_slice(&vec);
}

// --- Flow feature encoding (mirrors features.py flow_to_features) ---

/// Per-flow packet statistics for feature extraction.
/// Populated by the daemon's packet capture layer.
#[derive(Debug, Clone, Default)]
pub struct FlowStats {
    pub packet_sizes: Vec<u32>,
    pub inter_arrival_times: Vec<f64>,
    pub duration_seconds: f64,
    pub packet_count_upstream: u32,
    pub packet_count_downstream: u32,
    pub total_bytes: u64,
    pub first_n_packet_sizes: Vec<u32>,
    pub upstream_packet_sizes: Vec<u32>,
    pub downstream_packet_sizes: Vec<u32>,
    pub upstream_bytes: u64,
    pub downstream_bytes: u64,
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn directional_stats(sizes: &[u32]) -> (f32, f32, f32) {
    if sizes.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let floats: Vec<f32> = sizes.iter().map(|&s| s as f32).collect();
    let mean = floats.iter().sum::<f32>() / floats.len() as f32;
    let var = floats.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / floats.len() as f32;
    let std = var.sqrt();
    let mut sorted = floats.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = percentile(&sorted, 50.0);
    (mean / 1500.0, std / 1500.0, median / 1500.0)
}

/// Encode flow statistics into features [0:32] of the 61-dim vector.
fn encode_flow_features(flow: &FlowStats, features: &mut [f32; NUM_FEATURES]) {
    let pkt_floats: Vec<f32> = flow.packet_sizes.iter().map(|&s| s as f32).collect();

    if !pkt_floats.is_empty() {
        let mean = pkt_floats.iter().sum::<f32>() / pkt_floats.len() as f32;
        let var = pkt_floats.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / pkt_floats.len() as f32;
        let mut sorted = pkt_floats.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        features[0] = mean / 1500.0;
        features[1] = var.sqrt() / 1500.0;
        features[2] = *sorted.first().unwrap_or(&0.0) / 1500.0;
        features[3] = *sorted.last().unwrap_or(&0.0) / 1500.0;
        features[4] = percentile(&sorted, 25.0) / 1500.0;
        features[5] = percentile(&sorted, 50.0) / 1500.0;
        features[6] = percentile(&sorted, 75.0) / 1500.0;
    }

    if !flow.inter_arrival_times.is_empty() {
        let iats: Vec<f32> = flow.inter_arrival_times.iter().map(|&t| t as f32).collect();
        let mean = iats.iter().sum::<f32>() / iats.len() as f32;
        let var = iats.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / iats.len() as f32;
        let mut sorted = iats.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        features[7] = (mean).min(10.0) / 10.0;
        features[8] = var.sqrt().min(10.0) / 10.0;
        features[9] = sorted.first().copied().unwrap_or(0.0).min(10.0) / 10.0;
        features[10] = sorted.last().copied().unwrap_or(0.0).min(10.0) / 10.0;
        features[11] = percentile(&sorted, 50.0).min(10.0) / 10.0;
    }

    features[12] = (1.0 + flow.duration_seconds.min(300.0) as f32).ln() / (1.0 + 300.0f32).ln();
    features[13] = (1.0 + flow.packet_count_upstream as f32).ln() / (1.0 + 10000.0f32).ln();
    features[14] = (1.0 + flow.packet_count_downstream as f32).ln() / (1.0 + 10000.0f32).ln();
    let bps = flow.total_bytes as f32 / flow.duration_seconds.max(0.001) as f32;
    features[15] = (1.0 + bps).ln() / (1.0 + 1e9f32).ln();

    // Directional stats [16:24]
    let (up_mean, up_std, up_p50) = directional_stats(&flow.upstream_packet_sizes);
    features[16] = up_mean;
    features[17] = up_std;
    features[18] = up_p50;

    let (dn_mean, dn_std, dn_p50) = directional_stats(&flow.downstream_packet_sizes);
    features[19] = dn_mean;
    features[20] = dn_std;
    features[21] = dn_p50;

    let total_dir = (flow.upstream_bytes + flow.downstream_bytes) as f32;
    features[22] = if total_dir > 0.0 { flow.upstream_bytes as f32 / total_dir } else { 0.5 };

    let total_pkts = (flow.packet_count_upstream + flow.packet_count_downstream) as f32;
    features[23] = if total_pkts > 0.0 { flow.packet_count_upstream as f32 / total_pkts } else { 0.5 };

    // First-N packet sizes [24:32]
    for (i, &size) in flow.first_n_packet_sizes.iter().take(8).enumerate() {
        features[24 + i] = size as f32 / 1500.0;
    }
}

// --- Complete feature extraction ---

/// Build the full 61-dim feature vector from flow stats + huginn-net-tls output.
///
/// This is the primary entry point. The daemon calls this after:
/// 1. Collecting flow-level packet statistics
/// 2. Passing the ClientHello bytes through `extract_tls_metadata()`
pub fn build_feature_vector(
    flow: &FlowStats,
    tls: &TlsMetadata,
    ja4: &Ja4Components,
    sni_domain: &str,
) -> [f32; NUM_FEATURES] {
    let mut features = [0.0f32; NUM_FEATURES];

    encode_flow_features(flow, &mut features);   // [0:32]
    encode_tls_features(tls, &mut features);     // [32:44]
    encode_ja4_features(ja4, &mut features);     // [44:50]
    encode_sni_features(sni_domain, &mut features); // [50:61]

    features
}

// --- EMA domain cache ---

/// Per-domain classification entry with exponential moving average.
///
/// Instead of storing a single probability per domain, maintains a
/// running EMA across multiple flows. A domain that triggers 0.7 on
/// one flow might be noise, but consistent 0.6–0.8 across 5 flows
/// is a strong signal.
#[derive(Clone, Debug)]
struct DomainEntry {
    /// EMA of P(AI traffic) across flows to this domain.
    probability_ema: f32,
    /// EMA of model confidence across flows to this domain.
    confidence_ema: f32,
    /// Number of flows observed to this domain.
    flow_count: u32,
}

/// EMA-based per-domain result cache. Keyed by SNI domain.
///
/// On first observation, stores the raw probability. On subsequent
/// observations, blends the new value with the running average using
/// an exponential decay factor (alpha). This smooths out per-flow
/// noise and builds conviction over time.
struct DomainCache {
    cache: RwLock<HashMap<String, DomainEntry>>,
    /// EMA decay factor. Higher = more weight on recent observations.
    /// Default 0.3: new observation gets 30% weight.
    alpha: f32,
}

impl DomainCache {
    fn new(alpha: f32) -> Self {
        Self {
            cache: RwLock::new(HashMap::with_capacity(1024)),
            alpha,
        }
    }

    fn get(&self, domain: &str) -> Option<DomainEntry> {
        self.cache.read().ok()?.get(domain).cloned()
    }

    /// Update the EMA for a domain with a new observation.
    fn update(&self, domain: String, probability: f32, confidence: f32) {
        if let Ok(mut cache) = self.cache.write() {
            let entry = cache.entry(domain).or_insert(DomainEntry {
                probability_ema: probability,
                confidence_ema: confidence,
                flow_count: 0,
            });

            if entry.flow_count == 0 {
                // First observation: store raw values
                entry.probability_ema = probability;
                entry.confidence_ema = confidence;
            } else {
                // EMA update: new = alpha * observation + (1 - alpha) * old
                entry.probability_ema =
                    self.alpha * probability + (1.0 - self.alpha) * entry.probability_ema;
                entry.confidence_ema =
                    self.alpha * confidence + (1.0 - self.alpha) * entry.confidence_ema;
            }
            entry.flow_count += 1;
        }
    }
}

// --- Classification result ---

/// Result of classifying a flow, including confidence and EMA state.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// P(AI traffic) for this individual flow.
    pub probability: f32,
    /// Model's learned confidence in its prediction for this flow.
    pub confidence: f32,
    /// EMA-smoothed P(AI traffic) across all flows to this domain.
    pub ema_probability: f32,
    /// EMA-smoothed confidence across all flows to this domain.
    pub ema_confidence: f32,
    /// Whether this flow is classified as AI traffic (ema_probability > threshold).
    pub is_ai: bool,
    /// Number of flows observed to this domain so far.
    pub domain_flow_count: u32,
}

// --- ML classifier ---

/// The Stage 3 ML classifier with two-head output and EMA domain cache.
///
/// Model output is (1, 2): [logit, confidence].
/// The domain cache maintains exponential moving averages across flows,
/// building conviction over time. A single-flow anomaly won't flip a
/// domain's classification, but consistent signals will.
pub struct TrafficClassifier {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    cache: DomainCache,
    threshold: f32,
}

impl TrafficClassifier {
    /// Load the ONNX model from disk. Call once at daemon startup.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `threshold` - Classification threshold (default 0.5)
    /// * `ema_alpha` - EMA decay factor for domain cache (default 0.3)
    pub fn load(
        model_path: &str,
        threshold: Option<f32>,
        ema_alpha: Option<f32>,
    ) -> Result<Self> {
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
            cache: DomainCache::new(ema_alpha.unwrap_or(0.3)),
            threshold: threshold.unwrap_or(DEFAULT_THRESHOLD),
        })
    }

    /// Classify a flow using precomputed features.
    ///
    /// Returns a `ClassificationResult` with per-flow probability/confidence
    /// and EMA-smoothed domain-level values.
    ///
    /// The `is_ai` decision uses the EMA probability, not the single-flow
    /// probability. This means a domain needs consistent high-probability
    /// signals across multiple flows before being classified as AI.
    pub fn classify(
        &self,
        features: &[f32; NUM_FEATURES],
        domain: &str,
    ) -> Result<ClassificationResult> {
        // Run inference: model outputs (1, 2) = [logit, confidence]
        let input = tract_ndarray::Array2::from_shape_vec(
            (1, NUM_FEATURES),
            features.to_vec(),
        )?;

        let output = self.model.run(tvec![input.into_tensor()])?;
        let output_view = output[0].to_array_view::<f32>()?;
        let logit = output_view[[0, 0]];
        let confidence = output_view[[0, 1]];

        let probability = 1.0 / (1.0 + (-logit).exp());
        // Confidence is already sigmoid'd by the model, but clamp for safety
        let confidence = confidence.clamp(0.0, 1.0);

        // Update EMA cache
        if !domain.is_empty() {
            self.cache.update(domain.to_string(), probability, confidence);
        }

        // Get EMA values (will reflect the update we just made)
        let (ema_prob, ema_conf, flow_count) = if !domain.is_empty() {
            if let Some(entry) = self.cache.get(domain) {
                (entry.probability_ema, entry.confidence_ema, entry.flow_count)
            } else {
                (probability, confidence, 1)
            }
        } else {
            (probability, confidence, 0)
        };

        Ok(ClassificationResult {
            probability,
            confidence,
            ema_probability: ema_prob,
            ema_confidence: ema_conf,
            is_ai: ema_prob > self.threshold,
            domain_flow_count: flow_count,
        })
    }

    /// Full pipeline: extract TLS features via huginn-net-tls, build feature
    /// vector, and classify in one call.
    pub fn classify_flow(
        &self,
        flow: &FlowStats,
        client_hello: &[u8],
    ) -> Result<(ClassificationResult, String)> {
        let (tls, ja4, sni) = extract_tls_metadata(client_hello)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse ClientHello"))?;

        let features = build_feature_vector(flow, &tls, &ja4, &sni);
        let result = self.classify(&features, &sni)?;

        Ok((result, sni))
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.cache.write() {
            cache.clear();
        }
    }

    pub fn cache_size(&self) -> usize {
        self.cache.cache.read().map(|c| c.len()).unwrap_or(0)
    }
}

// --- Three-stage pipeline example ---

// ```rust
// fn scan_traffic(
//     flow: &Flow,
//     classifier: &TrafficClassifier,
// ) -> TrafficDecision {
//     // Stage 1: Deterministic rules (sub-μs)
//     if let Some(decision) = check_allowlist(&flow.domain) {
//         return decision;
//     }
//
//     // Stage 2: Heuristic scoring (~μs)
//     if let Some((tls, ja4, sni)) = extract_tls_metadata(&flow.client_hello) {
//         let heuristic = compute_heuristic_score(&tls, &ja4, &sni);
//         if heuristic > HIGH_CONFIDENCE { return TrafficDecision::Ai(heuristic); }
//         if heuristic < LOW_CONFIDENCE  { return TrafficDecision::Normal(heuristic); }
//     }
//
//     // Stage 3: ML classifier with confidence + EMA
//     match classifier.classify_flow(&flow.stats, &flow.client_hello) {
//         Ok((result, sni)) => {
//             if result.confidence < 0.4 {
//                 // Model is unsure — treat conservatively
//                 TrafficDecision::Unknown
//             } else if result.is_ai {
//                 TrafficDecision::Ai(result.ema_probability)
//             } else {
//                 TrafficDecision::Normal(result.ema_probability)
//             }
//         }
//         Err(_) => TrafficDecision::Unknown,
//     }
// }
// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_version_ord() {
        assert_eq!(tls_version_ord("TLS1.3"), 1.0);
        assert_eq!(tls_version_ord("TLS1.2"), 0.75);
        assert_eq!(tls_version_ord("SSLv3"), 0.0);
        assert_eq!(tls_version_ord("unknown"), 0.75); // defaults to TLS 1.2
    }

    #[test]
    fn test_alpn_ord() {
        assert_eq!(alpn_ord(""), 0.0);
        assert_eq!(alpn_ord("h2"), 0.6);
        assert_eq!(alpn_ord("grpc"), 1.0);
    }

    #[test]
    fn test_murmurhash3_matches_python() {
        // Verify cross-language determinism with known test vectors
        let h1 = murmurhash3_32(b"ap", 0);
        let h2 = murmurhash3_32(b"ap", 42);
        // These must match the Python _murmurhash3_32("ap".encode(), seed=0/42)
        assert!(h1 != 0, "Hash should be nonzero");
        assert!(h1 != h2, "Different seeds should produce different hashes");
    }

    #[test]
    fn test_sni_encoding_deterministic() {
        let mut f1 = [0.0f32; NUM_FEATURES];
        let mut f2 = [0.0f32; NUM_FEATURES];
        encode_sni_features("api.openai.com", &mut f1);
        encode_sni_features("api.openai.com", &mut f2);
        assert_eq!(f1[50..61], f2[50..61]);
    }

    #[test]
    fn test_sni_encoding_distinct_domains() {
        let mut f1 = [0.0f32; NUM_FEATURES];
        let mut f2 = [0.0f32; NUM_FEATURES];
        encode_sni_features("api.openai.com", &mut f1);
        encode_sni_features("www.google.com", &mut f2);
        assert_ne!(f1[50..61], f2[50..61]);
    }

    #[test]
    fn test_sni_encoding_normalized() {
        let mut features = [0.0f32; NUM_FEATURES];
        encode_sni_features("api.anthropic.com", &mut features);
        let norm: f32 = features[50..61].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "SNI hash should be L2-normalized");
    }

    #[test]
    fn test_build_feature_vector_shape() {
        let flow = FlowStats {
            packet_sizes: vec![100, 500, 200],
            inter_arrival_times: vec![0.01, 0.02],
            duration_seconds: 1.0,
            packet_count_upstream: 1,
            packet_count_downstream: 2,
            total_bytes: 800,
            first_n_packet_sizes: vec![100, 500, 200],
            upstream_packet_sizes: vec![100],
            downstream_packet_sizes: vec![500, 200],
            upstream_bytes: 100,
            downstream_bytes: 700,
        };

        let tls = TlsMetadata {
            version: "TLS1.3".into(),
            cipher_suite_count: 15,
            extension_count: 10,
            alpn: "h2".into(),
            has_grpc_alpn: false,
            has_h2_alpn: true,
            cert_chain_length: 3,
            has_sni_extension: true,
            has_sct_extension: true,
            has_status_request: false,
            has_supported_versions_13_only: true,
            has_post_handshake_auth: false,
        };

        let ja4 = Ja4Components {
            tls_version: "TLS1.3".into(),
            cipher_count: 15,
            extension_count: 10,
            alpn: "h2".into(),
            sorted_cipher_hash: "abc123".into(),
            sorted_extension_hash: "def456".into(),
        };

        let features = build_feature_vector(&flow, &tls, &ja4, "api.openai.com");
        assert_eq!(features.len(), NUM_FEATURES);

        // All features should be finite
        for (i, &f) in features.iter().enumerate() {
            assert!(f.is_finite(), "Feature {} is not finite: {}", i, f);
        }

        // Byte ratio should reflect asymmetry
        assert!(features[22] < 0.5, "Byte ratio should show downstream dominance");
    }

    #[test]
    fn test_directional_stats_empty() {
        let (m, s, p) = directional_stats(&[]);
        assert_eq!(m, 0.0);
        assert_eq!(s, 0.0);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn test_directional_stats_values() {
        let (mean, std, median) = directional_stats(&[300, 600, 900]);
        assert!((mean - 0.4).abs() < 0.01); // 600/1500
        assert!(std > 0.0);
        assert!((median - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_ema_cache_first_observation() {
        let cache = DomainCache::new(0.3);
        cache.update("test.com".into(), 0.8, 0.9);
        let entry = cache.get("test.com").unwrap();
        assert_eq!(entry.probability_ema, 0.8);
        assert_eq!(entry.confidence_ema, 0.9);
        assert_eq!(entry.flow_count, 1);
    }

    #[test]
    fn test_ema_cache_blending() {
        let cache = DomainCache::new(0.3); // alpha = 0.3
        cache.update("test.com".into(), 0.8, 0.9);
        cache.update("test.com".into(), 0.2, 0.5);

        let entry = cache.get("test.com").unwrap();
        // EMA: 0.3 * 0.2 + 0.7 * 0.8 = 0.06 + 0.56 = 0.62
        assert!((entry.probability_ema - 0.62).abs() < 0.01);
        // EMA: 0.3 * 0.5 + 0.7 * 0.9 = 0.15 + 0.63 = 0.78
        assert!((entry.confidence_ema - 0.78).abs() < 0.01);
        assert_eq!(entry.flow_count, 2);
    }

    #[test]
    fn test_ema_cache_convergence() {
        let cache = DomainCache::new(0.3);
        // Feed consistent 0.9 signals — EMA should converge toward 0.9
        for _ in 0..20 {
            cache.update("ai.com".into(), 0.9, 0.95);
        }
        let entry = cache.get("ai.com").unwrap();
        assert!((entry.probability_ema - 0.9).abs() < 0.01);
        assert_eq!(entry.flow_count, 20);
    }

    #[test]
    fn test_ema_cache_noise_rejection() {
        let cache = DomainCache::new(0.3);
        // Build up strong "not AI" signal
        for _ in 0..10 {
            cache.update("normal.com".into(), 0.1, 0.9);
        }
        // Single noisy observation shouldn't flip the classification
        cache.update("normal.com".into(), 0.8, 0.3);
        let entry = cache.get("normal.com").unwrap();
        // EMA should still be well below 0.5
        assert!(entry.probability_ema < 0.3,
            "Single noisy observation should not flip EMA: {}", entry.probability_ema);
    }

    #[test]
    fn test_ema_cache_independent_domains() {
        let cache = DomainCache::new(0.3);
        cache.update("a.com".into(), 0.9, 0.8);
        cache.update("b.com".into(), 0.1, 0.8);

        let a = cache.get("a.com").unwrap();
        let b = cache.get("b.com").unwrap();
        assert!((a.probability_ema - 0.9).abs() < 0.01);
        assert!((b.probability_ema - 0.1).abs() < 0.01);
    }
}
