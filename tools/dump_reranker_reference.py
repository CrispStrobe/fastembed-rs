#!/usr/bin/env python3
"""Offline PyTorch reference dumper for reranker variants.

Companion to `tools/dump_reference.py` (which does embedders).  For each
(query, doc) pair in a fixed multilingual test set, computes the scalar
relevance score from the upstream HF cross-encoder (FP32 PyTorch) and
writes a safetensors fixture that `tests/reranker_parity.rs` loads to
gate the ONNX variant via Spearman correlation + per-group ranking
order.

Why fixture-based and not exact-checksum: cross-encoder logits drift
across CPU microarchitectures the same way embedder outputs do.  Spearman
is rank-correlation — invariant to monotone transformations, so a
quantized model can have offset/scale shifts in absolute scores yet
still preserve ranking and pass.

Why PyTorch reference (not our FP32 ONNX): an ONNX export bug would be
invisible if we validated ONNX-against-ONNX.

Test-set design (mirrors scripts/reranker_diff.py):
  4 query groups (English + German), 4 docs each → 16 pairs total.
  Within each group, the FIRST doc is expected to be the most relevant.

Fixture format (safetensors):
  Tensors:
    scores       f32 [N_pairs]   reference scalar relevance scores
  Metadata (__metadata__ JSON):
    model_repo           HF repo id of the reference cross-encoder
    revision             pinned commit hash
    threshold_spearman   minimum Spearman the Rust test asserts (default 0.9)
    groups               JSON array of {query, docs[], expected_top1_pair_idx}
    score_convention     "logit_1class" or "logit_for_minus_against"
    notes                free-text caveats

Example:
  python tools/dump_reranker_reference.py \\
    --repo Alibaba-NLP/gte-reranker-modernbert-base \\
    --threshold-spearman 0.9 \\
    --output tests/fixtures/reranker__GteReranker.safetensors

  # Generative-style reranker (Qwen3 chat template, "Yes" token logit):
  python tools/dump_reranker_reference.py \\
    --repo zeroentropy/zerank-1-small \\
    --score-convention generative_yes_token \\
    --threshold-spearman 0.9 \\
    --output tests/fixtures/reranker__ZerankSmall.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# Same probe set as scripts/reranker_diff.py — keep these in sync.
TEST_GROUPS: list[tuple[str, list[str]]] = [
    ("What city is the capital of France?", [
        "Paris is the capital of France and a major European city.",
        "Berlin is the capital of Germany.",
        "ONNX Runtime can execute exported neural networks efficiently on CPU.",
        "A stock market index tracks the price of selected securities.",
    ]),
    ("How can neural embedding models be deployed?", [
        "ONNX Runtime can execute exported neural networks efficiently on CPU.",
        "Neural networks are inspired by the structure of the human brain.",
        "Paris is the capital of France and a major European city.",
        "Climate change affects biodiversity through habitat loss and stress.",
    ]),
    ("Welche Auswirkungen hat der Klimawandel auf Biodiversitaet?", [
        "Der Klimawandel beeinflusst die biologische Vielfalt.",
        "Climate change affects biodiversity through habitat loss and stress.",
        "Rust prevents many memory safety bugs at compile time.",
        "Paris is the capital of France and a major European city.",
    ]),
    ("Was ist Mitose?", [
        "Bei der Mitose teilt sich eine Zelle in zwei genetisch identische Tochterzellen.",
        "Mitosis is the process by which a cell replicates its chromosomes and divides into two identical daughter cells.",
        "The Eiffel Tower stands 330 metres tall in central Paris.",
        "ONNX Runtime can execute exported neural networks efficiently on CPU.",
    ]),
]


def import_deps():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from huggingface_hub import snapshot_download
    from safetensors.numpy import save_file
    return torch, AutoModelForSequenceClassification, AutoTokenizer, snapshot_download, save_file


def build_pairs(
    groups: list[tuple[str, list[str]]],
) -> tuple[list[tuple[str, str]], list[int], list[int]]:
    """Return (pairs, group_id_per_pair, expected_top1_pair_idx_per_group)."""
    pairs: list[tuple[str, str]] = []
    group_id: list[int] = []
    expected_top1: list[int] = []
    for g_idx, (q, docs) in enumerate(groups):
        first = len(pairs)
        for d in docs:
            pairs.append((q, d))
            group_id.append(g_idx)
        expected_top1.append(first)
    return pairs, group_id, expected_top1


def hf_scores_cross_encoder(
    snapshot: Path,
    pairs: list[tuple[str, str]],
    max_length: int,
    trust_remote_code: bool,
) -> tuple[np.ndarray, str]:
    """Score each pair via AutoModelForSequenceClassification.  Returns
    (scores, score_convention) where convention is "logit_1class" or
    "logit_for_minus_against" depending on output dim."""
    torch, AutoModelForSeqCls, AutoTokenizer, _, _ = import_deps()
    tok = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=trust_remote_code)
    model = AutoModelForSeqCls.from_pretrained(
        str(snapshot),
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    enc = tok(
        [p[0] for p in pairs], [p[1] for p in pairs],
        padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits.detach().cpu().numpy().astype(np.float32)
    if logits.ndim == 2 and logits.shape[1] == 1:
        scores = logits[:, 0]
        convention = "logit_1class"
    elif logits.ndim == 2 and logits.shape[1] == 2:
        # [for, against]: score = for - against
        scores = logits[:, 0] - logits[:, 1]
        convention = "logit_for_minus_against"
    else:
        scores = logits.reshape(logits.shape[0], -1)[:, 0]
        convention = "logit_first_col"
    return scores, convention


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", required=True, help="HF repo id of the reference reranker")
    p.add_argument("--revision", default=None, help="Pinned commit hash")
    p.add_argument("--cache-dir", default="/Volumes/backups/ai/huggingface-hub")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-length", type=int, default=256)

    p.add_argument("--threshold-spearman", type=float, default=0.9,
                   help="Min Spearman the Rust test asserts.  Default 0.9 — the operational "
                        "shipping bar.  Encoder rerankers typically pass 0.95+, decoder INT8 "
                        "rerankers may pass at 0.93–0.97.")
    p.add_argument("--notes", default="",
                   help="Free-text caveat stored in fixture metadata.")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to write the safetensors fixture.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch, _, _, snapshot_download, save_file = import_deps()

    snap_kwargs = dict(
        repo_id=args.repo,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if args.revision:
        snap_kwargs["revision"] = args.revision
    snap = Path(snapshot_download(**snap_kwargs))
    print(f"HF snapshot: {snap}")

    refs_main = (snap.parent.parent / "refs" / "main")
    if args.revision:
        commit = args.revision
    elif refs_main.exists():
        commit = refs_main.read_text().strip()
    else:
        commit = snap.name

    pairs, group_id, expected_top1 = build_pairs(TEST_GROUPS)
    print(f"Pairs: {len(pairs)} across {max(group_id)+1} groups")

    scores, convention = hf_scores_cross_encoder(
        snap, pairs, args.max_length, args.trust_remote_code
    )
    print(f"Reference scores: shape={scores.shape}, range=[{scores.min():.3f}, {scores.max():.3f}]")
    print(f"Score convention: {convention}")

    # For each group, report which pair PyTorch picked as best.  This
    # confirms the reference itself is sane before shipping the fixture.
    n_groups = max(group_id) + 1
    ref_top1: list[int] = []
    for g in range(n_groups):
        idx_in_group = [i for i, gid in enumerate(group_id) if gid == g]
        best = max(idx_in_group, key=lambda i: scores[i])
        ref_top1.append(best)
    print(f"Reference top1 per group: {ref_top1}  expected: {expected_top1}  "
          f"match={ref_top1 == expected_top1}")
    if ref_top1 != expected_top1:
        print("  WARN: reference does not pick the expected top-1 in every group. "
              "The fixture will still gate ONNX vs reference, but the test set may need review.",
              flush=True)

    metadata = {
        "model_repo": args.repo,
        "revision": commit,
        "threshold_spearman": str(args.threshold_spearman),
        "max_length": str(args.max_length),
        "score_convention": convention,
        "groups": json.dumps([
            {
                "query": q,
                "docs": docs,
                "expected_top1_pair_idx": expected_top1[g_idx],
                "ref_top1_pair_idx": ref_top1[g_idx],
            }
            for g_idx, (q, docs) in enumerate(TEST_GROUPS)
        ]),
        "group_id_per_pair": json.dumps(group_id),
        "notes": args.notes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"scores": scores}, str(args.output), metadata=metadata)
    size = args.output.stat().st_size
    print(f"\nWrote {args.output} ({size / 1024:.2f} KB)")
    print(f"  pinned to {args.repo}@{commit}")
    print(f"  spearman threshold >= {args.threshold_spearman}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
