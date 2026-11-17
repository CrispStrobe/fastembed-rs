# German GECToR: conditional native ONNX backend

Date: 2026-11-17. Branch: `feat/german-gector-proofing`.
Status: roadmap only; engineering fallback after CrispEmbed Q4_K, not license-approved for shipping.
Base: `d1500cdf248fc5447fb723f38e7cf3ee21b30487` from probe/ort-rc12.

The [canonical Quds roadmap](https://github.com/CrispStrobe/quds_office_suite/blob/feat/german-gector-proofing/docs/roadmaps/german-gector-proofing.md) owns scope, model provenance,
decoder behavior, backend decision and acceptance gates.

## Bounded scope

The bounded Dart round missed the complete-check latency budget. CrispEmbed
Q4_K is the next qualification lane; activate this Rust fallback if that path
fails its quality, performance or packaging gates, and this backend can satisfy
the strict permissive-license boundary below. Provide a small dedicated
grammar/session or prepared-input API using the existing ort/tokenizers stack. Reuse pinned file-based model loading but keep
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
completed successfully; retain the local baseline's earlier queued-at-capture
status unchanged. The optimized follow-up is recorded below.

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

## Measured decision after the optimized follow-up

[Run 44062093617](https://github.com/CrispStrobe/quds_office_suite/actions/runs/44062093617)
at Quds `69d153c0f81596af894e48e21880f0957b963994` and
Dart `b10258682d459847b63e012eaaba58a528e93956` passed all 30 cases /
40 saved passes and the explicit runtime-experiment integrity check.

On that run's Intel CPU, the same ONNX INT4 candidate took 15.8/17.1 ms
median/p95 per forward in native ORT, versus 463/546 ms in optimized Dart.
Native complete reference correction had 33.2 ms p95, excluding application
guards. The original hosted run used a different AMD CPU, so cross-run changes
are not an isolated hardware-controlled code comparison.

The next selected implementation/qualification target is the preferred
CrispEmbed/ggml mixed Q4_K path. Retain this prepared-input native ORT plan as
the fallback; it has promising measured engine behavior but still requires
the compatible API-24 runtime, its own parity run and an actual Dart FFI bridge.
No Rust model/session, wrapper or shipping application has been implemented
by this roadmap update.

## Shipping-license boundary found during final qualification review

The measured Python ORT 1.23.2 control demonstrates native engine behavior;
it does not approve that distribution for the application. ORT has an MIT
top-level license, but the pinned
[1.23.2 third-party notices](https://github.com/microsoft/onnxruntime/blob/a83fc4d58cb48eb68890dd689f94f28288cf2278/ThirdPartyNotices.txt)
include Eigen under MPL-2.0. That falls outside this feature's strict
MIT/Apache/BSD/similar permissive allowlist. Do not rely on the Rust wrapper's
license or ORT's top-level license to approve the complete native dependency
closure.

Native ORT therefore remains a development/performance oracle and a conditional
engineering fallback, not an approved shipping choice. Before activating it,
qualify the exact API-24-compatible binary's compiled dependencies and applicable
notices; the measured 1.23.2 wheel is neither a version-matched substitute nor
license clearance for another build. No compatible, fully permissive binary
has been demonstrated by this work. Do not relax the user's license constraint
to make the fallback available. CrispEmbed's controlled ggml/Q4_K target stays
the preferred next lane, with its own artifact-level license qualification.
