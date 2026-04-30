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

- [x] validate F2LLM `int8_pt` (per-tensor dynamic) — **FAIL**, cos_min=0.249,
      worse than per-channel `int8` (0.304). Dynamic INT8 weight-only quant
      is fundamentally inadequate for F2LLM regardless of granularity.
- [~] run + validate F2LLM `int8_static` (QDQ + text calibration) — running
- [ ] if `int8_static` also fails: try INT4 MatMulNBits or recommend dropping
      `F2LlmV2_0_6BInt8`
- [ ] decide whether `F2LlmV2_0_6BInt8` should be dropped, replaced, or kept;
      record decision in `fastembed-rs-wip/LEARNINGS.md`

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

For each: load FP32 (or whatever the upstream original is) from HF, get
ground-truth embeddings, then test every ONNX variant we ship.

Embedding models:

- [ ] PixieRuneV1                  (cls)
- [ ] PixieRuneV1Q                 (cls, INT8)
- [ ] PixieRuneV1Int4              (cls, INT4 MatMulNBits)
- [ ] PixieRuneV1Int4Full          (cls, INT4 + INT8 Gather)
- [ ] OctenEmbedding0_6BFp32       (last_token)
- [ ] OctenEmbedding0_6BInt4       (last_token, INT4 MatMulNBits)
- [ ] OctenEmbedding0_6BInt8Full   (last_token, INT8 static)
- [ ] OctenEmbedding0_6BInt4Full   (last_token, INT4 + INT8 Gather)
- [~] F2LlmV2_0_6BFp32             — PASS (cached + freshly exported)
- [!] F2LlmV2_0_6BInt8             — FAIL (cos_min 0.304, per-channel dyn MatMul)
- [ ] F2LlmV2_0_6BInt4
- [ ] F2LlmV2_0_6BInt8Full
- [ ] JinaEmbeddingsV3             (mean / task-specific)
- [ ] JinaEmbeddingsV5Nano         (pre_pooled `sentence_embedding`)
- [ ] JinaEmbeddingsV5Small        (pre_pooled `sentence_embedding`)
- [ ] HarrierOSSV1_270M            (cls/mean — verify)
- [ ] HarrierOSSV1_270MQ
- [ ] SnowflakeArcticEmbedMV2      (cls)
- [ ] GteModernBertBase            (cls)
- [ ] GteModernBertBaseQ           (cls, INT8)
- [ ] GteModernBertBaseQ4F16       (cls, INT4 + FP16)

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

- [ ] for every FAIL: decide drop / requantize / fix-pooling; mark in
      `fastembed-rs-wip/LEARNINGS.md` with the cosine numbers
- [ ] update PR descriptions of PR-a/b/c with the validation table
- [ ] open follow-up issues for any FAIL we don't fix in this round
