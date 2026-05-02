//! Cosine-parity CI gate for shipped models.
//!
//! Each fixture under `tests/fixtures/<Variant>.safetensors` is a frozen
//! PyTorch reference produced offline by `tools/dump_reference.py`:
//!   - `embeddings`  f32 [N, dim]  unnormalised reference vectors
//!   - `input_ids` / `attention_mask`  int64  (informational)
//!   - metadata: `texts` (JSON array), `threshold`, `model_repo`, `revision`,
//!     `pooling`, `notes`
//!
//! For every entry in `FIXTURES`, this test:
//!   1. Reads the fixture (skipping the entry if the file is absent locally).
//!   2. Loads the model through fastembed-rs's public `TextEmbedding::try_new`.
//!   3. Embeds the same texts that were dumped to the fixture.
//!   4. Computes per-row cosine vs the reference embeddings.
//!   5. Asserts `cos_min >= threshold` from the fixture metadata.
//!
//! Why fixture-based and not exact checksums: ORT INT8/INT4 accumulation
//! varies across CPU microarchitectures, so exact element-wise equality is
//! not portable across CI runners. The cosine of two vectors computed on
//! the same machine (CI vs precomputed reference values) IS portable
//! within the threshold bands set per variant.
//!
//! Why PyTorch reference (not our FP32 ONNX): an ONNX export bug would be
//! invisible if we validated ONNX-against-ONNX. PyTorch is gold.
//!
//! To regenerate a fixture:
//!   python tools/dump_reference.py --repo <hf-id> --pooling <mode> \
//!     --threshold <bar> --output tests/fixtures/<Variant>.safetensors
//!
//! Suggested thresholds:
//!   FP32                  0.999
//!   FP16                  0.99
//!   Encoder INT8 / Q      0.95
//!   Decoder INT8 / SmoothQuant   0.90
//!   INT4 / INT4Full              0.90

#![cfg(feature = "hf-hub")]

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// Registry: every fixture file → which `EmbeddingModel` variant runs against it.
const FIXTURES: &[(EmbeddingModel, &str)] = &[(
    EmbeddingModel::AllMiniLML6V2,
    "tests/fixtures/AllMiniLML6V2.safetensors",
)];

/// Parse an f32 tensor stored as little-endian bytes into a flat `Vec<f32>`.
/// (Safetensors stores tensor data as packed bytes; the dtype tells us how to
/// interpret them. The existing codebase uses the same chunks_exact(4) pattern
/// in `src/sparse_text_embedding/bgem3_weights.rs`.)
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Returns true iff the model is present in the local HF cache.
/// Mirrors the offline-skip pattern used by `tests/text-embeddings.rs`.
fn model_in_cache(model_code: &str) -> bool {
    if std::env::var("HF_HUB_OFFLINE").as_deref() != Ok("1") {
        return true; // online mode: just attempt the download
    }
    let dir_name = format!("models--{}", model_code.replace('/', "--"));
    let cache_dirs =
        std::env::var("FASTEMBED_CACHE_DIR").unwrap_or_else(|_| ".fastembed_cache".into());
    cache_dirs.split(':').filter(|s| !s.is_empty()).any(|dir| {
        let refs_main = Path::new(dir).join(&dir_name).join("refs/main");
        if let Ok(hash) = std::fs::read_to_string(&refs_main) {
            Path::new(dir)
                .join(&dir_name)
                .join("snapshots")
                .join(hash.trim())
                .exists()
        } else {
            false
        }
    })
}

#[test]
fn cosine_parity_against_pytorch_reference() {
    let mut failures: Vec<String> = Vec::new();
    let mut ran = 0;

    for (model, fixture_path) in FIXTURES {
        eprintln!("\n== {model:?} (fixture: {fixture_path}) ==");

        // Skip silently if fixture wasn't generated for this checkout.
        let bytes = match std::fs::read(fixture_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  SKIP: cannot read fixture: {e}");
                continue;
            }
        };
        let (header_size, metadata) = match SafeTensors::read_metadata(&bytes) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{model:?}: SafeTensors header parse failed: {e}"));
                continue;
            }
        };
        let _ = header_size;
        let meta_kv: HashMap<String, String> = metadata
            .metadata()
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let texts_json = match meta_kv.get("texts") {
            Some(s) => s.clone(),
            None => {
                failures.push(format!("{model:?}: fixture missing `texts` metadata"));
                continue;
            }
        };
        let texts: Vec<String> = match serde_json::from_str(&texts_json) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{model:?}: cannot parse texts JSON: {e}"));
                continue;
            }
        };

        let threshold: f32 = meta_kv
            .get("threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.95);

        // Sanity-check that the fixture matches the variant being tested.
        let info = TextEmbedding::get_model_info(&model).expect("model info");
        if let Some(repo) = meta_kv.get("model_repo") {
            if repo != &info.model_code {
                eprintln!(
                    "  WARN: fixture model_repo={repo} but EmbeddingModel maps to {} — \
                     fixture may be stale",
                    info.model_code
                );
            }
        }

        // Load the reference embeddings.
        let st = match SafeTensors::deserialize(&bytes) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{model:?}: SafeTensors deserialize failed: {e}"));
                continue;
            }
        };
        let ref_view = match st.tensor("embeddings") {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{model:?}: missing `embeddings` tensor: {e}"));
                continue;
            }
        };
        let ref_shape = ref_view.shape();
        if ref_shape.len() != 2 {
            failures.push(format!(
                "{model:?}: expected 2D embeddings, got shape {:?}",
                ref_shape
            ));
            continue;
        }
        let n_inputs = ref_shape[0];
        let dim = ref_shape[1];
        let ref_flat = bytes_to_f32(ref_view.data());
        if ref_flat.len() != n_inputs * dim {
            failures.push(format!(
                "{model:?}: embeddings byte-count mismatch (got {}, expected {})",
                ref_flat.len(),
                n_inputs * dim
            ));
            continue;
        }

        if texts.len() != n_inputs {
            failures.push(format!(
                "{model:?}: fixture has {} texts but {} embedding rows",
                texts.len(),
                n_inputs
            ));
            continue;
        }

        // Skip gracefully if model files aren't in the local cache.
        let offline = std::env::var("HF_HUB_OFFLINE").as_deref() == Ok("1");
        if offline && !model_in_cache(&info.model_code) {
            eprintln!("  SKIP: not in local cache (HF_HUB_OFFLINE=1)");
            continue;
        }

        let mut embedder = match TextEmbedding::try_new(InitOptions::new(model.clone())) {
            Ok(e) => e,
            Err(e) if offline => {
                eprintln!("  SKIP: load failed offline: {e}");
                continue;
            }
            Err(e) => {
                failures.push(format!("{model:?}: try_new failed: {e}"));
                continue;
            }
        };

        let texts_ref: Vec<&str> = texts.iter().map(String::as_str).collect();
        let got: Vec<Vec<f32>> = match embedder.embed(texts_ref, None) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{model:?}: embed failed: {e}"));
                continue;
            }
        };

        if got.len() != n_inputs {
            failures.push(format!(
                "{model:?}: model returned {} embeddings, expected {}",
                got.len(),
                n_inputs
            ));
            continue;
        }
        if got[0].len() != dim {
            failures.push(format!(
                "{model:?}: dim mismatch: got {}, fixture has {}",
                got[0].len(),
                dim
            ));
            continue;
        }

        // Per-row cosine vs reference.
        let mut cos_min = f32::INFINITY;
        let mut cos_mean = 0.0f32;
        let mut per_row: Vec<f32> = Vec::with_capacity(n_inputs);
        for i in 0..n_inputs {
            let r = &ref_flat[i * dim..(i + 1) * dim];
            let g = &got[i];
            let c = cosine(r, g);
            cos_min = cos_min.min(c);
            cos_mean += c;
            per_row.push(c);
        }
        cos_mean /= n_inputs as f32;

        eprintln!(
            "  cos_min={:.6} cos_mean={:.6}  threshold>={:.4}  rows={:?}",
            cos_min, cos_mean, threshold, per_row
        );

        if cos_min < threshold {
            failures.push(format!(
                "{model:?}: cos_min={cos_min:.6} below threshold {threshold:.4} \
                 (rows={per_row:?})"
            ));
            continue;
        }
        ran += 1;
    }

    if ran == 0 && failures.is_empty() {
        eprintln!("\nNote: no fixtures were exercised (none present locally and/or all skipped).");
    }

    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
        panic!("{} cosine-parity assertion(s) failed", failures.len());
    }
}
