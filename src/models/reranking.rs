use std::{fmt::Display, str::FromStr};

use crate::RerankerModelInfo;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum RerankerModel {
    /// BAAI/bge-reranker-base
    #[default]
    BGERerankerBase,
    /// rozgo/bge-reranker-v2-m3
    BGERerankerV2M3,
    /// jinaai/jina-reranker-v1-turbo-en
    JINARerankerV1TurboEn,
    /// jinaai/jina-reranker-v2-base-multilingual — FP32
    JINARerankerV2BaseMultiligual,
    /// jinaai/jina-reranker-v2-base-multilingual — INT8 quantized
    JINARerankerV2BaseMultilingualInt8,
    /// jinaai/jina-reranker-v2-base-multilingual — FP16 (`onnx/model_fp16.onnx`).
    /// Uses graph optimisations (`SimplifiedLayerNormFusion` with
    /// `InsertedPrecisionFreeCast`) that require ORT >= 1.23
    /// (`ort = 2.0.0-rc.12+`).  Validated via reranker-parity at
    /// `spearman_threshold=0.9`.
    JINARerankerV2BaseMultilingualFp16,
    // ── mixedbread-ai mxbai-rerank ────────────────────────────────────────────
    /// mixedbread-ai/mxbai-rerank-xsmall-v1 — 33M, English, 512 tokens
    MxbaiRerankXsmallV1,
    /// mixedbread-ai/mxbai-rerank-xsmall-v1 — INT8-quantized
    MxbaiRerankXsmallV1Q,
    /// mixedbread-ai/mxbai-rerank-base-v1 — 86M, English, 512 tokens
    MxbaiRerankBaseV1,
    /// mixedbread-ai/mxbai-rerank-base-v1 — INT8-quantized
    MxbaiRerankBaseV1Q,
    /// mixedbread-ai/mxbai-rerank-large-v1 — 560M, English, 512 tokens
    MxbaiRerankLargeV1,
    /// mixedbread-ai/mxbai-rerank-large-v1 — INT8-quantized
    MxbaiRerankLargeV1Q,
    // ── cross-encoder/ms-marco MiniLM ─────────────────────────────────────────
    /// cross-encoder/ms-marco-MiniLM-L-6-v2 — 22M, English, fast
    MsMarcoMiniLML6V2,
    /// cross-encoder/ms-marco-MiniLM-L-12-v2 — 33M, English, high quality
    MsMarcoMiniLML12V2,
    // ── nvidia/llama-nemotron-rerank-1b-v2 ────────────────────────────────────
    /// nvidia/llama-nemotron-rerank-1b-v2 — 1B, multilingual, LLaMA-3.2 bidirectional (FP32)
    LlamaNemotronRerank1BV2,
    /// nvidia/llama-nemotron-rerank-1b-v2 — INT4 MatMul + INT8 Gather (~832 MB).
    /// Block-wise INT4 weights preserve dynamic range better than per-channel
    /// vanilla INT8 (Spearman 0.944 per LEARNINGS Phase 7).  Requires ORT
    /// >= 1.23 (`ort = 2.0.0-rc.12+`).
    LlamaNemotronRerank1BV2Int4Full,
    // ── Alibaba-NLP/gte-reranker-modernbert-base ──────────────────────────────
    /// Alibaba-NLP/gte-reranker-modernbert-base — 149M, English, 8192 tokens (FP32, 596 MB)
    GteRerankerModernBertBase,
    /// Alibaba-NLP/gte-reranker-modernbert-base — INT8 quantized (150 MB).
    /// English-only: per-group Spearman vs FP32 = 1.0 on English queries,
    /// drops to ~0.8 on non-English queries.  ModernBERT is English-trained,
    /// so this is consistent with the underlying model's design.
    GteRerankerModernBertBaseQ,
    /// Alibaba-NLP/gte-reranker-modernbert-base — Q4F16 mixed-precision (140 MB).
    /// Uses graph optimisations that require ORT >= 1.23
    /// (`ort = 2.0.0-rc.12+`).  Validated via reranker-parity.
    GteRerankerModernBertBaseQ4F16,
    // ── zeroentropy/zerank-1-small (Qwen3 1.7B reranker, multilingual) ────────
    /// zeroentropy/zerank-1-small — 1.7B Qwen3 reranker, multilingual (FP32 ONNX, ~3.4 GB; batch-broadcast patched)
    ZerankSmall,
    /// zeroentropy/zerank-1-small — INT8 quantized (~1.7 GB; batch-broadcast patched).
    /// Per LEARNINGS Phase 7: Spearman 0.965 vs PyTorch on the multilingual
    /// probe set (English-clean, slight ranking drift on some non-English queries).
    /// Requires ORT >= 1.23 (`ort = 2.0.0-rc.12+`).
    ZerankSmallInt8,
    /// zeroentropy/zerank-1-small — INT4 MatMulNBits (~860 MB; batch-broadcast patched).
    /// Per LEARNINGS Phase 7: Spearman 0.935 vs PyTorch on the multilingual
    /// probe set.  Requires ORT >= 1.23 (`ort = 2.0.0-rc.12+`).
    ZerankSmallInt4,
}

pub fn reranker_model_list() -> Vec<RerankerModelInfo> {
    let reranker_model_list = vec![
        RerankerModelInfo {
            model: RerankerModel::BGERerankerBase,
            description: String::from("reranker model for English and Chinese"),
            model_code: String::from("BAAI/bge-reranker-base"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::BGERerankerV2M3,
            description: String::from("reranker model for multilingual"),
            model_code: String::from("rozgo/bge-reranker-v2-m3"),
            model_file: String::from("model.onnx"),
            additional_files: vec![String::from("model.onnx.data")],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV1TurboEn,
            description: String::from("reranker model for English"),
            model_code: String::from("jinaai/jina-reranker-v1-turbo-en"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV2BaseMultiligual,
            description: String::from(
                "Jina reranker v2, multilingual — 278M, 1024 tokens, XLM-RoBERTa (FP32)",
            ),
            model_code: String::from("jinaai/jina-reranker-v2-base-multilingual"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV2BaseMultilingualInt8,
            description: String::from(
                "Jina reranker v2, multilingual — 278M, 1024 tokens, XLM-RoBERTa (INT8)",
            ),
            model_code: String::from("jinaai/jina-reranker-v2-base-multilingual"),
            model_file: String::from("onnx/model_int8.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::JINARerankerV2BaseMultilingualFp16,
            description: String::from(
                "Jina reranker v2, multilingual — 278M, 1024 tokens, XLM-RoBERTa (FP16). \
                 Requires ORT >= 1.23 (`ort = 2.0.0-rc.12+`).",
            ),
            model_code: String::from("jinaai/jina-reranker-v2-base-multilingual"),
            model_file: String::from("onnx/model_fp16.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        // ── mxbai-rerank ─────────────────────────────────────────────────────
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankXsmallV1,
            description: String::from(
                "mxbai-rerank-xsmall-v1 — 33M, English, DeBERTa-v3 cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-xsmall-v1"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankXsmallV1Q,
            description: String::from(
                "mxbai-rerank-xsmall-v1 INT8 — 33M, English, DeBERTa-v3 cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-xsmall-v1"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankBaseV1,
            description: String::from(
                "mxbai-rerank-base-v1 — 86M, English, DeBERTa-v3 cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-base-v1"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankBaseV1Q,
            description: String::from(
                "mxbai-rerank-base-v1 INT8 — 86M, English, DeBERTa-v3 cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-base-v1"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankLargeV1,
            description: String::from(
                "mxbai-rerank-large-v1 — 560M, English, DeBERTa-v3-large cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-large-v1"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::MxbaiRerankLargeV1Q,
            description: String::from(
                "mxbai-rerank-large-v1 INT8 — 560M, English, DeBERTa-v3-large cross-encoder",
            ),
            model_code: String::from("mixedbread-ai/mxbai-rerank-large-v1"),
            model_file: String::from("onnx/model_quantized.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        // ── ms-marco MiniLM ───────────────────────────────────────────────────
        RerankerModelInfo {
            model: RerankerModel::MsMarcoMiniLML6V2,
            description: String::from(
                "ms-marco-MiniLM-L-6-v2 — 22M, English, fast BERT cross-encoder",
            ),
            model_code: String::from("cross-encoder/ms-marco-MiniLM-L-6-v2"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::MsMarcoMiniLML12V2,
            description: String::from("ms-marco-MiniLM-L-12-v2 — 33M, English, BERT cross-encoder"),
            model_code: String::from("cross-encoder/ms-marco-MiniLM-L-12-v2"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        // ── nvidia/llama-nemotron-rerank-1b-v2 ───────────────────────────────
        RerankerModelInfo {
            model: RerankerModel::LlamaNemotronRerank1BV2,
            description: String::from(
                "nvidia/llama-nemotron-rerank-1b-v2 — 1B LLaMA-3.2 bidirectional reranker, multilingual (FP32, 4.6 GB)",
            ),
            model_code: String::from("cstr/llama-nemotron-rerank-1b-v2-ONNX"),
            model_file: String::from("model.onnx"),
            additional_files: vec![String::from("model.onnx_data")],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::LlamaNemotronRerank1BV2Int4Full,
            description: String::from(
                "nvidia/llama-nemotron-rerank-1b-v2 — INT4 MatMul + INT8 Gather (~832 MB). \
                 Block-wise INT4 weights preserve dynamic range; Spearman 0.944 vs PyTorch \
                 (LEARNINGS Phase 7).  Requires ORT >= 1.23 (`ort = 2.0.0-rc.12+`).",
            ),
            model_code: String::from("cstr/llama-nemotron-rerank-1b-v2-ONNX"),
            model_file: String::from("model_int4_full.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        // ── Alibaba-NLP/gte-reranker-modernbert-base ─────────────────────────
        RerankerModelInfo {
            model: RerankerModel::GteRerankerModernBertBase,
            description: String::from(
                "gte-reranker-modernbert-base — 149M, English, 8192 tokens, ModernBERT cross-encoder (FP32, 596 MB)",
            ),
            model_code: String::from("Alibaba-NLP/gte-reranker-modernbert-base"),
            model_file: String::from("onnx/model.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::GteRerankerModernBertBaseQ,
            description: String::from(
                "gte-reranker-modernbert-base — 149M, English, 8192 tokens, ModernBERT cross-encoder (INT8, 150 MB)",
            ),
            model_code: String::from("Alibaba-NLP/gte-reranker-modernbert-base"),
            model_file: String::from("onnx/model_int8.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        RerankerModelInfo {
            model: RerankerModel::GteRerankerModernBertBaseQ4F16,
            description: String::from(
                "gte-reranker-modernbert-base — 149M, English, 8192 tokens, ModernBERT cross-encoder (Q4F16, ~140 MB). \
                 Requires ORT >= 1.23 (`ort = 2.0.0-rc.12+`).",
            ),
            model_code: String::from("Alibaba-NLP/gte-reranker-modernbert-base"),
            model_file: String::from("onnx/model_q4f16.onnx"),
            additional_files: vec![],
            prompt_template: None,
        },
        // ── zeroentropy/zerank-1-small ────────────────────────────────────────
        // The originally-published export had a hardcoded batch=1 in the
        // attention-mask broadcast (the And kernel can't broadcast {1,1,T,T}
        // against {B,1,T,T}).  This variant is the patched re-upload (Expand
        // + Shape inserted before the And, stale value_info entries dropped)
        // and works at any batch size.
        RerankerModelInfo {
            model: RerankerModel::ZerankSmall,
            description: String::from(
                "zeroentropy/zerank-1-small — 1.7B Qwen3 reranker, multilingual (FP32 ONNX, ~3.4 GB; \
                 batch-broadcast patched)",
            ),
            model_code: String::from("cstr/zerank-1-small-ONNX"),
            model_file: String::from("model.onnx"),
            additional_files: vec![String::from("model.onnx_data")],
            prompt_template: Some(String::from(
                "<|im_start|>user\nQuery: {query}\nDocument: {doc}\nRelevant:<|im_end|>\n<|im_start|>assistant\n",
            )),
        },
        RerankerModelInfo {
            model: RerankerModel::ZerankSmallInt8,
            description: String::from(
                "zeroentropy/zerank-1-small — INT8 (~1.7 GB; batch-broadcast patched). \
                 Spearman 0.965 vs PyTorch (LEARNINGS Phase 7).  Requires ORT >= 1.23 \
                 (`ort = 2.0.0-rc.12+`).",
            ),
            model_code: String::from("cstr/zerank-1-small-ONNX"),
            model_file: String::from("model_int8.onnx"),
            additional_files: vec![String::from("model_int8.onnx_data")],
            prompt_template: Some(String::from(
                "<|im_start|>user\nQuery: {query}\nDocument: {doc}\nRelevant:<|im_end|>\n<|im_start|>assistant\n",
            )),
        },
        RerankerModelInfo {
            model: RerankerModel::ZerankSmallInt4,
            description: String::from(
                "zeroentropy/zerank-1-small — INT4 MatMulNBits (~860 MB; batch-broadcast patched). \
                 Spearman 0.935 vs PyTorch (LEARNINGS Phase 7).  Requires ORT >= 1.23 \
                 (`ort = 2.0.0-rc.12+`).",
            ),
            model_code: String::from("cstr/zerank-1-small-ONNX"),
            model_file: String::from("model_int4_full.onnx"),
            additional_files: vec![],
            prompt_template: Some(String::from(
                "<|im_start|>user\nQuery: {query}\nDocument: {doc}\nRelevant:<|im_end|>\n<|im_start|>assistant\n",
            )),
        },
    ];
    reranker_model_list
}

impl Display for RerankerModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let model_info = reranker_model_list()
            .into_iter()
            .find(|model| model.model == *self)
            .ok_or(std::fmt::Error)?;
        write!(f, "{}", model_info.model_code)
    }
}

impl FromStr for RerankerModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        reranker_model_list()
            .into_iter()
            .find(|m| m.model_code.eq_ignore_ascii_case(s))
            .map(|m| m.model)
            .ok_or_else(|| format!("Unknown reranker model: {s}"))
    }
}

impl TryFrom<String> for RerankerModel {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}
