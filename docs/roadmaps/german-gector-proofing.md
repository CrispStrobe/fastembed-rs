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

## Native follow-up after the two-case smoke — 17 November 2026

The [Quds baseline record](https://github.com/CrispStrobe/quds_office_suite/blob/feat/german-gector-proofing/docs/validation/german-gector/2026-11-17-local-smoke/README.md)
pins evaluation source `6bd03365ba3b146586148658b0e03998b36bdc9d` and
Dart source `0b49ca47fa056e6fc122ffe0192a9abb05627622`. Native Python ORT
1.23.2 measured roughly 26-30 ms per F32 forward and 45 ms per INT4 forward;
the INT4 complete two-pass corrections were approximately 85-125 ms on these
two short cases. This suggests a useful native fallback, but it is not a Rust
benchmark or a qualified Dart FFI bridge. The selected Swiss false edit remains.
The full [hosted qualification run](https://github.com/CrispStrobe/quds_office_suite/actions/runs/44053903485)
is executing with final evidence review pending; retain the baseline's earlier
queued-at-capture status unchanged.

### Version and prepared-input boundary

At feature source `99d21fbc830e3dbaaafff3d60a0864097b1b795b`,
[Cargo.toml](https://github.com/CrispStrobe/fastembed-rs/blob/99d21fbc830e3dbaaafff3d60a0864097b1b795b/Cargo.toml)
pins `ort = 2.0.0-rc.12` with `api-24`. The measured oracle used ORT 1.23.2.
Pin and verify a matching native runtime and repeat same-input parity; do not
silently load the measured wheel library under a different API contract.

The [existing user-defined embedding constructor](https://github.com/CrispStrobe/fastembed-rs/blob/99d21fbc830e3dbaaafff3d60a0864097b1b795b/src/text_embedding/impl.rs)
uses `available_parallelism()` for thread count and its string transform path
constructs tokenization internally. Raw output access does not make that path a
prepared-input API; pooled `embed()` output is unsuitable.

If this fallback is activated, add a dedicated prepared token-classification
session, with a small module exported from
[src/lib.rs](https://github.com/CrispStrobe/fastembed-rs/blob/99d21fbc830e3dbaaafff3d60a0864097b1b795b/src/lib.rs).
Load a verified local model, accept exact int64 [1,T] IDs/masks, explicitly set
two intra-op threads and sequential execution, and validate both named float
outputs [1,T,5000] and [1,T,2]. Keep one owned persistent session with serialized
calls and clear result lifetimes. Reuse shared Dart alignment/decoding/guards.

The current crate does not provide the required Dart C ABI packaging. A separate
small cdylib bridge, ownership/error contract, Dart wrapper and target-platform
native-library packaging remain explicit work after the prepared-session proof.
Record native binary/API identity, sources and model hashes; qualify the actual
bridge rather than transferring the Python control's latency claim.

The already measured ONNX INT4 candidate is 158,365,857 bytes, SHA-256
`6c92deb88469e2fce2ddded5159014200d8681a7853722714224539a48285d70`.
It can be reused after artifact verification without another model conversion.
It is not Q4_K. Keep the preferred CrispEmbed Q4_K experiment and this fallback
ordered by the backend decision, rather than implementing both concurrently.
This update records source feasibility only; native Rust/FFI implementation,
numerical parity, performance and platform qualification are still pending.
