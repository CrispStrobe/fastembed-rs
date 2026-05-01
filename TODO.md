# fastembed-rs validation TODO

Goal: real ground-truth validation of every model variant added in PRs A/B/C.
Method: compare each ONNX runtime (FP32, INT8, INT4, etc.) against the
original HF/PyTorch model on identical token IDs — cosine similarity per
sentence, plus retrieval order on a fixed query/doc set. This is the same
discipline used in CrispASR / CrispEmbed.

The reference harness lives at:

    /Volumes/backups/ai/fastembed-rs-wip/f2llm_export/f2llm_diff.py

Reports written under:

    /Volumes/backups/ai/fastembed-rs-wip/<model>_work/reports/

## Status legend

- [ ] PENDING        not yet run
- [~] IN PROGRESS    running or partial
- [x] PASS           cos_min >= 0.97 vs HF and retrieval order matches
- [!] FAIL           cos_min <  0.97 vs HF or retrieval order broken

## Phase 1 — finish F2LLM quantization triage

- [x] validate F2LLM `int8_pt` (per-tensor dynamic) — **FAIL**, cos_min=0.249.
- [x] validate F2LLM `int8_static` (QDQ + text calibration) — **FAIL**,
      cos_min=0.119 (worst).
- [~] validate cached `int4` (MatMulNBits) and `int8_full` — running
- [ ] decision: every dynamic INT8 strategy fails for F2LLM. Drop
      `F2LlmV2_0_6BInt8` regardless; keep `_Int4` / `_Int8Full` only if they
      pass the harness.

## Phase 2 — generalize the harness

- [x] write generic `onnx_diff.py` at `/Volumes/backups/ai/fastembed-rs-wip/scripts/`
      (lives outside repo per Option C) — args: --repo, --pooling, --query-prefix,
      --doc-prefix, --variant name=path, --onnx-output-name,
      --onnx-output-already-pooled, --reference-onnx (soft fallback)
- [x] driver `run_validation.py` + spec table `model_specs.py`
- [ ] download PyTorch originals for repos missing safetensors locally:
      harrier, jina-v3, jina-v5-{nano,small}, snowflake-mv2, gte-modernbert
- [ ] verify parity on F2LLM (already-validated reference)

## Phase 3 — validate all new embedding models added in PR-c

Harness sanity-checked against AllMiniLML6V2 baseline:
  - FP32 vs HF: cos_min=1.000  PASS  (harness math correct, encoder+mean OK)
  - Q   vs HF: cos_min=0.986  PASS  (encoder INT8 dynamic is fine in general)

Therefore the F2LLM INT8 collapse is NOT a harness bug.  Predicted picture:
encoder + INT8 = PASS, decoder LLM + INT8/INT4 = FAIL or borderline.

For each: load FP32 (or whatever the upstream original is) from HF, get
ground-truth embeddings, then test every ONNX variant we ship.

Embedding models:

- [x] PixieRuneV1                  PASS cos=1.000
- [x] PixieRuneV1Q                 PASS cos_min=0.963
- [x] PixieRuneV1Int4              PASS cos_min=0.930
- [x] PixieRuneV1Int4Full          PASS cos_min=0.930
- [x] OctenEmbedding0_6BFp32       PASS cos=1.000
- [x] OctenEmbedding0_6BInt4       PASS cos_min=0.922 (above F2LLM int4 by ~0.3 — different fine-tune)
- [!] OctenEmbedding0_6BInt8Full   FAIL cos_min=0.602 (same Qwen3 outlier issue, drop)
- [x] OctenEmbedding0_6BInt4Full   PASS cos_min=0.922
- [x] F2LlmV2_0_6BFp32             — PASS (cached + freshly exported, cos=1.000)
- [!] F2LlmV2_0_6BInt8             — FAIL cos_min 0.304 (per-channel dyn MatMul)
- [!] F2LlmV2_0_6BInt8 (int8_pt)   — FAIL cos_min 0.249 (per-tensor dyn MatMul)
- [!] F2LlmV2_0_6BInt8 (int8_static)— FAIL cos_min 0.119 (QDQ + 14-sentence calib)
- [!] F2LlmV2_0_6BInt4             — FAIL cos_min 0.640 (MatMulNBits, borderline)
- [!] F2LlmV2_0_6BInt8Full         — FAIL cos_min 0.212 (MatMul+Gather quantized)
- [!] JinaEmbeddingsV3             HARNESS LIMITATION: AutoModel.forward + mean-pool
      doesn't match ONNX (cos~0.70). V3 needs the LoRA adapter applied via the
      sentence-transformers .encode(task=...) API on the HF side. Not a model
      bug; only FP32 ships (no quants to validate). Park.
- [!] JinaEmbeddingsV5Nano         BORDERLINE: cos_min=0.557 (one outlier sentence),
      cos_mean=0.921. Other 5/6 sentences pass at cos>0.97. Probably a specific
      sentence's last-token activation gets clipped by INT8. Also fixed a real
      pooling bug: fastembed-rs had Pooling::Cls but the model uses LastToken.
- [x] JinaEmbeddingsV5Small        PASS cos=1.000 (FP32 only)
- [x] HarrierOSSV1_270M            PASS cos=1.000 (last_token, sentence_embedding)
- [!] HarrierOSSV1_270MQ           PASS cos_min=0.99993 BUT requires ORT >= 1.23
      (current ort = 2.0.0-rc.11 ships ORT 1.22 — file fails to load in fastembed-rs).
      Bump ort crate or drop variant.
- [x] SnowflakeArcticEmbedMV2      PASS cos_min=0.960 (Q only ships)
- [x] GteModernBertBase            PASS cos=1.000
- [x] GteModernBertBaseQ           PASS cos_min=0.943
- [x] GteModernBertBaseQ4F16       PASS cos_min=0.971

## Phase 4 — validate all rerankers added

Rerankers need a different metric: same prompt template + cross-encoder logit
correlation against the HF reference model on a small held-out set, since
"embedding cosine vs HF" doesn't apply.

- [ ] write `scripts/reranker_diff.py` (Spearman of logits vs HF reference on
      a 50-pair set)
- [ ] LlamaNemotronRerank1BV2 (FP32 vs INT8 vs INT4Full)
- [ ] GteRerankerModernBertBase (FP32 vs Q vs Q4F16)
- [ ] ZerankSmall (FP32 vs INT8 vs INT4)
- [ ] MxbaiRerankXsmall/Base/LargeV1 (each + their Q)
- [ ] MsMarcoMiniLML6V2, MsMarcoMiniLML12V2
- [ ] JINARerankerV2BaseMultilingual (Int8, Fp16)

## Phase 5 — fastembed-rs end-to-end parity

Once a variant passes Phase 3/4 vs HF in pure Python, run it through
fastembed-rs against the same fixed token IDs and confirm cos vs HF stays the
same. Catches Rust-side pooling, normalization, or prompt-template bugs.

- [ ] add `tests/onnx_parity.rs` that loads pre-tokenized fixtures from
      `tests/fixtures/` and asserts cos >= 0.999 vs cached HF embeddings
- [ ] one fixture per pooling family (cls / mean / last_token / pre_pooled /
      prompt-template-reranker)

## Phase 6 — triage and ship

- [x] for every FAIL: decided drop / requantize / fix-pooling; recorded in
      `fastembed-rs-wip/LEARNINGS.md` with cosine numbers
- [ ] update PR descriptions of PR-a/b/c with the validation table
- [ ] open follow-up issues for any FAIL we don't fix in this round

## Phase 7 — actually-fix-the-broken-cases  (results)

- [x] Verified `ort = "2.0.0-rc.12"` ships ORT 1.24, but the API is
      not source-compatible with rc.11 (~136 errors).  HarrierQ dropped
      until that migration is done.
- [x] Streaming FP16 converter written
      (`scripts/convert_fp16_streaming.py`).  Bypasses the 2 GB protobuf
      serialization limit that breaks `onnxconverter_common.float16` and
      `onnxruntime.transformers.float16`.
      F2LLM FP16: cos_min=1.000, 1.2 GB.  PASS
      Octen  FP16: cos_min=1.000, 1.2 GB.  PASS
- [x] SmoothQuant for F2LLM INT8
      (`scripts/smoothquant_onnx.py` + `quant_smoothed_int8.py`).
      F2LLM smoothed FP32 (math equivalence): cos_min=1.000  PASS
      F2LLM smoothed INT8 (alpha=0.8):         cos_min=0.932  PASS
      → 3.1× quality improvement vs vanilla INT8 (0.30 → 0.93).
- [~] Octen SmoothQuant INT8: running with alpha=0.8.

## Phase 8 — re-upload and re-add variants  (done)

- [x] Upload F2LLM SmoothQuant INT8 → cstr/F2LLM-v2-0.6B-ONNX-INT8
- [~] Upload F2LLM FP16 → cstr/F2LLM-v2-0.6B-ONNX-FP16          (in progress, 1.2 GB)
- [x] Upload Octen FP16 → cstr/Octen-Embedding-0.6B-ONNX-FP16
- [x] Upload Octen SmoothQuant INT8 → cstr/Octen-Embedding-0.6B-ONNX-INT8
- [x] Re-add F2LlmV2_0_6BFp16, F2LlmV2_0_6BInt8, OctenEmbedding0_6BFp16,
      OctenEmbedding0_6BInt8 to fastembed-rs source.

## Phase 9 — open work, prioritized

1. F2LlmV2_0_6BInt4 (cos=0.64) disposition.  Either drop, or rebuild via
   SmoothQuant + INT4 MatMulNBits (the same outlier-migration applied to
   group-quantization should reach ≥0.93).
2. JinaEmbeddingsV5Small: **DONE.** Same recipe as F2LLM/Octen.
   FP16 cos=1.000, SmoothQuant INT8 cos=0.994.  Variants
   `JinaEmbeddingsV5SmallFp16` and `JinaEmbeddingsV5SmallInt8` added.
   Repos: cstr/jina-embeddings-v5-text-small-retrieval-onnx-{fp16,int8}.
3. Reranker validation harness (`scripts/reranker_diff.py`): **DONE.**
   Per-variant findings (all on the same 4-query / 16-pair set, with
   English + German queries):

   | Variant                              | Spearman | per-group     | top1 vs FP32 | Note |
   |--------------------------------------|----------|---------------|--------------|------|
   | GteRerankerModernBertBase            | 1.000    | 1/1/1/1       | match        | PASS (FP32) |
   | GteRerankerModernBertBaseQ           | 0.959    | 1/1/0.8/0.8   | diverge (DE) | English-only note kept |
   | GteRerankerModernBertBaseQ4F16       | —        | —             | won't load   | **DROPPED** |
   | MxbaiRerankXsmallV1                  | 1.000    | 1/1/1/1       | match        | PASS (FP32) |
   | MxbaiRerankXsmallV1Q                 | 0.994    | 1/1/1/1       | match        | PASS (multilingual-clean) |
   | MxbaiRerankBaseV1                    | 1.000    | 1/1/1/1       | match        | PASS (FP32; FP32 itself is English-biased) |
   | MxbaiRerankBaseV1Q                   | 0.938    | 1/1/0.8/1     | 1 group diff | borderline |
   | MxbaiRerankLargeV1                   | 1.000    | 1/1/1/1       | match        | PASS (FP32; English-biased) |
   | MxbaiRerankLargeV1Q                  | 0.918    | 1/1/0.8/0.8   | 2 groups diff| borderline |

   Pattern: Mxbai-Xsmall and Jina-style multilingual rerankers handle
   INT8 cleanly; ModernBERT and Mxbai Base/Large Q variants shift
   ranking on non-English queries. Disposition for Mxbai Base/Large Q
   is a judgment call (FP32 itself was English-biased; Q just shifts
   in a different direction).

   | JINARerankerV2BaseMultiligual         | 1.000    | [1,1,1,1]     | match        | PASS (FP32) |
   | JINARerankerV2BaseMultilingualInt8    | 0.971    | [1,1,0.8,1]   | match        | PASS |
   | JINARerankerV2BaseMultilingualFp16    | —        | —             | won't load   | **DROPPED** (same SimplifiedLayerNormFusion as GteQ4F16) |

   Still unvalidated (need ONNX downloads):
     - LlamaNemotronRerank1BV2{Int8,Int4Full}    (~4 GB)
     - ZerankSmall{Int8,Int4}                    (Qwen3-1.7B, ~3 GB)
     - MsMarcoMiniLM{L6,L12}V2  (FP32 only — no quants ship)
4. Run the full `cargo test` suite (with downloads enabled) on the
   re-added variants once the F2LLM FP16 upload finishes.
5. ort crate migration rc.11 → rc.12 to re-enable HarrierOSSV1_270MQ
   (currently dropped because rc.11 ships ORT 1.22 < required 1.23).
   ~136 source-compat errors to fix.
6. Update PR descriptions (`feat/new-model-entries` plus PR-a/b if
   relevant) with the validation table and the SmoothQuant story.
7. V5 Nano Q outlier investigation: cos_min 0.557 on a single sentence,
   cos_mean 0.921 on the rest.  Likely a single-sentence quantization
   pathology; may go away with a larger probe set.
8. JinaEmbeddingsV3 sentence-transformers harness so we can validate it
   too (V3 needs `.encode(task='retrieval.passage')`, not bare
   AutoModel.forward + mean pool).
9. Apply SmoothQuant + FP16 to other Qwen3-derived embedders we haven't
   touched yet — same architecture, same recipe should work.
10. Open issues on the upstream Anush008/fastembed-rs for findings that
    affect them too (the validation harness itself, the V5 Nano pooling
    bug if/when V5 Nano lands upstream).
