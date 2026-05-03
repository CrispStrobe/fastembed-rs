//! Spearman + ranking-order CI gate for shipped reranker variants.
//!
//! Companion to `tests/cosine_parity.rs` (which gates embedders on cosine).
//! Cross-encoder rerankers produce a scalar relevance score per (query, doc)
//! pair; quantization can introduce monotone-ish offset/scale shifts that
//! preserve ranking even when absolute logits drift.  Spearman is rank
//! correlation — invariant to monotone transformations — so it's the right
//! metric here.
//!
//! Each fixture under `tests/fixtures/reranker__<Variant>.safetensors` is a
//! frozen PyTorch reference produced offline by
//! `tools/dump_reranker_reference.py`:
//!   - `scores`  f32 [N_pairs]   reference scalar relevance scores
//!   - metadata: `model_repo`, `revision`, `threshold_spearman`, `groups`
//!     (JSON: per-query groups + expected_top1_pair_idx), `score_convention`,
//!     `notes`
//!
//! For every entry in `FIXTURES`, this test:
//!   1. Reads the fixture (skipping if absent locally).
//!   2. Loads the reranker through `TextRerank::try_new`.
//!   3. For each query group, calls `model.rerank(query, docs, ..)` and
//!      captures scalar score per doc in ORIGINAL doc order (via
//!      `RerankResult.index`).
//!   4. Concatenates per-group scores in pair order to match the reference.
//!   5. Asserts:
//!        - Spearman(reference_scores, model_scores) >= threshold_spearman
//!        - Per-group top-1 pair index matches the fixture's
//!          `expected_top1_pair_idx`
//!
//! To regenerate a fixture:
//!   python tools/dump_reranker_reference.py \
//!     --repo <hf-id> --threshold-spearman <bar> \
//!     --output tests/fixtures/reranker__<Variant>.safetensors

#![cfg(feature = "hf-hub")]

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// Registry: every fixture file → which `RerankerModel` variant runs against it.
///
/// Multiple quantized siblings of one cross-encoder can share a single
/// fixture (the reference is the upstream PyTorch model; each sibling is
/// run independently against that reference).  The optional third tuple
/// element overrides the threshold stored in the fixture metadata.
const FIXTURES: &[(RerankerModel, &str, Option<f32>)] = &[(
    RerankerModel::MxbaiRerankXsmallV1,
    "tests/fixtures/reranker__MxbaiRerankXsmallV1.safetensors",
    None,
)];

#[derive(serde::Deserialize)]
struct Group {
    query: String,
    docs: Vec<String>,
    expected_top1_pair_idx: usize,
    #[serde(default)]
    #[allow(dead_code)]
    ref_top1_pair_idx: usize,
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Spearman rank correlation: convert both inputs to ranks, then compute
/// Pearson on the ranks.  Ties get fractional ranks (averaged).
fn spearman(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let ra = ranks(a);
    let rb = ranks(b);
    pearson(&ra, &rb)
}

/// Fractional ranks (1-based; ties averaged).
fn ranks(xs: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&i, &j| {
        xs[i]
            .partial_cmp(&xs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = vec![0.0_f32; xs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && xs[idx[j]] == xs[idx[i]] {
            j += 1;
        }
        // average rank for the tied block [i..j)
        let avg = (i + j + 1) as f32 / 2.0; // 1-based: ranks i+1..=j averaged
        for k in i..j {
            out[idx[k]] = avg;
        }
        i = j;
    }
    out
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut num = 0.0_f32;
    let mut da = 0.0_f32;
    let mut db = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let xa = x - mean_a;
        let yb = y - mean_b;
        num += xa * yb;
        da += xa * xa;
        db += yb * yb;
    }
    let denom = (da.sqrt()) * (db.sqrt());
    if denom < 1e-12 {
        0.0
    } else {
        num / denom
    }
}

/// Mirrors the cache check used in `tests/text-embeddings.rs`.
fn model_in_cache(model_code: &str) -> bool {
    if std::env::var("HF_HUB_OFFLINE").as_deref() != Ok("1") {
        return true;
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
fn reranker_parity_against_pytorch_reference() {
    let mut failures: Vec<String> = Vec::new();

    for (model, fixture_path, threshold_override) in FIXTURES {
        eprintln!("\n== {model:?} (fixture: {fixture_path}) ==");

        let bytes = match std::fs::read(fixture_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  SKIP: cannot read fixture: {e}");
                continue;
            }
        };
        let (_, header) = match SafeTensors::read_metadata(&bytes) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{model:?}: SafeTensors header parse failed: {e}"));
                continue;
            }
        };
        let meta_kv: HashMap<String, String> = header
            .metadata()
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let groups_json = match meta_kv.get("groups") {
            Some(s) => s.clone(),
            None => {
                failures.push(format!("{model:?}: fixture missing `groups` metadata"));
                continue;
            }
        };
        let groups: Vec<Group> = match serde_json::from_str(&groups_json) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{model:?}: cannot parse groups JSON: {e}"));
                continue;
            }
        };

        let threshold: f32 = threshold_override.unwrap_or_else(|| {
            meta_kv
                .get("threshold_spearman")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.9)
        });

        let info = TextRerank::get_model_info(model);
        if let Some(repo) = meta_kv.get("model_repo") {
            if repo != &info.model_code {
                eprintln!(
                    "  WARN: fixture model_repo={repo} but RerankerModel maps to {} \
                     — fixture may be stale",
                    info.model_code
                );
            }
        }

        let st = match SafeTensors::deserialize(&bytes) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{model:?}: SafeTensors deserialize failed: {e}"));
                continue;
            }
        };
        let scores_view = match st.tensor("scores") {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{model:?}: missing `scores` tensor: {e}"));
                continue;
            }
        };
        let ref_scores = bytes_to_f32(scores_view.data());
        let total_pairs: usize = groups.iter().map(|g| g.docs.len()).sum();
        if ref_scores.len() != total_pairs {
            failures.push(format!(
                "{model:?}: reference has {} scores but groups imply {} pairs",
                ref_scores.len(),
                total_pairs
            ));
            continue;
        }

        let offline = std::env::var("HF_HUB_OFFLINE").as_deref() == Ok("1");
        if offline && !model_in_cache(&info.model_code) {
            eprintln!("  SKIP: not in local cache (HF_HUB_OFFLINE=1)");
            continue;
        }

        let mut reranker = match TextRerank::try_new(RerankInitOptions::new(model.clone())) {
            Ok(r) => r,
            Err(e) if offline => {
                eprintln!("  SKIP: load failed offline: {e}");
                continue;
            }
            Err(e) => {
                failures.push(format!("{model:?}: try_new failed: {e}"));
                continue;
            }
        };

        // Score each group; flatten back to pair-index order to match reference.
        let mut got_scores: Vec<f32> = Vec::with_capacity(total_pairs);
        let mut group_top1_pair_idx: Vec<usize> = Vec::with_capacity(groups.len());
        let mut pair_offset = 0usize;
        for g in &groups {
            // Compiler can't always infer the generic `S` here; explicit `&str` annotation.
            let docs_str: Vec<&str> = g.docs.iter().map(String::as_str).collect();
            let results = match reranker.rerank::<&str>(&g.query, docs_str, false, None) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!(
                        "{model:?}: rerank failed for group {:?}: {e}",
                        g.query
                    ));
                    got_scores.resize(got_scores.len() + g.docs.len(), 0.0);
                    pair_offset += g.docs.len();
                    group_top1_pair_idx.push(usize::MAX);
                    continue;
                }
            };
            // Re-order scores back to original doc order (RerankResult.index).
            let mut scores_in_order = vec![0.0_f32; g.docs.len()];
            for r in &results {
                scores_in_order[r.index] = r.score;
            }
            // Best in this group: highest score → its pair index = pair_offset + best_doc_idx
            let best_doc_idx = (0..g.docs.len())
                .max_by(|&a, &b| {
                    scores_in_order[a]
                        .partial_cmp(&scores_in_order[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            group_top1_pair_idx.push(pair_offset + best_doc_idx);
            got_scores.extend_from_slice(&scores_in_order);
            pair_offset += g.docs.len();
        }

        let sr = spearman(&ref_scores, &got_scores);
        let expected: Vec<usize> = groups.iter().map(|g| g.expected_top1_pair_idx).collect();
        let top1_match = group_top1_pair_idx == expected;

        eprintln!(
            "  spearman={sr:.4}  threshold>={threshold:.4}  \
             top1_per_group got={group_top1_pair_idx:?} expected={expected:?} match={top1_match}"
        );

        if sr < threshold {
            failures.push(format!(
                "{model:?}: spearman={sr:.4} below threshold {threshold:.4}"
            ));
            continue;
        }
        if !top1_match {
            failures.push(format!(
                "{model:?}: top1 per group mismatch: got {group_top1_pair_idx:?}, \
                 expected {expected:?}"
            ));
            continue;
        }
    }

    if !failures.is_empty() {
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
        panic!("{} reranker-parity assertion(s) failed", failures.len());
    }
}
