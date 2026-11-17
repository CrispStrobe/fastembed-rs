# German GECToR: conditional native ONNX backend

Date: 2026-11-17. Branch: `feat/german-gector-proofing`.
Status: roadmap only; implementation conditional on Dart qualification.
Base: `d1500cdf248fc5447fb723f38e7cf3ee21b30487` from probe/ort-rc12.

The [canonical Quds roadmap](https://github.com/CrispStrobe/quds_office_suite/blob/feat/german-gector-proofing/docs/roadmaps/german-gector-proofing.md) owns scope, model provenance,
decoder behavior, backend decision and acceptance gates.

## Bounded scope

If onnx_runtime_dart fails the bounded resource/performance qualification,
provide a small dedicated grammar/session or prepared-input API using the
existing ort/tokenizers stack. Reuse pinned file-based model loading but keep
a grammar model separate from pooled/normalized embedding model registration.

- Accept exact token IDs/masks and preserve word/subword alignment.
- Request both logits_labels and logits_d and validate names/dtypes/shapes.
- Bound thread counts, batching, context ownership and cancellation between
  calls; the reference operates on one unpadded sentence at a time.
- Keep shared decoding, guards, locale policy and UTF-16 source mapping in the
  Quds Dart package.
- Reuse exact frozen Python/Dart fixtures and require final correction parity
  in addition to numerical comparisons.
- Record the native ORT binary identity and exact source/model/fixture hashes.

## Quantization

The user prefers Q4_K, which belongs to the conditional CrispEmbed/ggml route.
This backend runs ONNX; any MatMulNBits INT4 or other quantized candidate is
separate and must be named and measured honestly. Preserve both classifier
heads and qualify decoded quality, latency, artifact size and memory together.
F32 is a reference, not an implicitly approved shipping precision.

## Current boundary

No generic embedding API change, model download, native runtime upgrade, model
publication, release or main merge follows from this roadmap. The fork default
branch currently uses ort2.0.0-rc.12; pin exact tested versions. Keep changes
recoverable on this feature branch and update the canonical evidence record
when the native lane is activated.
