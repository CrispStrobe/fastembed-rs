#!/usr/bin/env python3
"""Offline PyTorch reference dumper for fastembed-rs cosine-parity CI tests.

Loads an upstream HF model in PyTorch (firsthand ground truth, not our ONNX
re-export), runs it on a fixed canonical text set, and writes a small
safetensors fixture that the in-tree Rust test (`tests/cosine_parity.rs`)
loads and compares against. The Rust test never touches PyTorch — it just
runs the ONNX-under-test through fastembed-rs's public API and asserts
cosine similarity vs the precomputed reference embeddings.

Why PyTorch and not our FP32 ONNX as reference: an ONNX export bug would
be invisible if we validated ONNX-against-ONNX. PyTorch is gold.

Why fixture-based and not "compute reference in CI": CI cannot afford to
download multi-GB PyTorch checkpoints + run them. The fixture is ~10–50 KB
per model.

Cross-CPU portability: the cosine of two vectors computed on the same
machine (CI's own ONNX run vs the precomputed reference values from the
fixture) is portable across CPU microarchitectures within the threshold
bands set per variant — typically <1e-6 drift for FP32, <1e-3 for INT8.

Fixture format (safetensors):
  Tensors:
    input_ids        int64 [N, max_seq_len]   — informational, for debugging
    attention_mask   int64 [N, max_seq_len]   — informational
    embeddings       f32   [N, dim]           — the unnormalized reference
  Metadata (__metadata__ JSON):
    model_repo       HF repo id
    revision         commit hash (pinned)
    pooling          cls | mean | last_token | pre_pooled
    threshold        f32 — min cos_min the test will assert
    texts            JSON array — the canonical input strings
    notes            free-text — methodology caveats

Example:
  python tools/dump_reference.py \\
    --repo sentence-transformers/all-MiniLM-L6-v2 \\
    --pooling mean \\
    --threshold 0.999 \\
    --output tests/fixtures/AllMiniLML6V2.safetensors

  # Decoder LLM with last-token pooling and asymmetric prefix:
  python tools/dump_reference.py \\
    --repo codefuse-ai/F2LLM-v2-0.6B \\
    --pooling last_token \\
    --query-prefix "Instruct: ...\\nQuery: " \\
    --threshold 0.99 \\
    --output tests/fixtures/F2LlmV2_0_6BFp32.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


# Canonical text set. Six items spanning short/long, EN/DE/multilingual.
# Kept in sync with scripts/onnx_diff.py's TEXTS so reports cross-compare.
TEXTS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can be deployed efficiently using ONNX Runtime.",
    "Paris is the capital of France and a major European city.",
    "Rust focuses on memory safety and zero-cost abstractions.",
    "Der Klimawandel beeinflusst die biologische Vielfalt.",
    "Retrieval embeddings should place matching questions and passages nearby.",
]


def import_deps():
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from huggingface_hub import snapshot_download
    from safetensors.numpy import save_file
    return torch, AutoConfig, AutoModel, AutoTokenizer, snapshot_download, save_file


def parse_extra_input(spec: str) -> tuple[str, np.ndarray]:
    """Parse '--extra-input task_id=1' -> ('task_id', 0-D int64 array)."""
    import ast
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--extra-input expects name=value, got: {spec!r}")
    name, _, raw = spec.partition("=")
    try:
        val = ast.literal_eval(raw.strip())
    except Exception:
        raise argparse.ArgumentTypeError(f"Could not parse {raw!r} as Python literal")
    if isinstance(val, (int, np.integer)):
        return name.strip(), np.array(val, dtype=np.int64)
    if isinstance(val, (float, np.floating)):
        return name.strip(), np.array(val, dtype=np.float32)
    if isinstance(val, list):
        return name.strip(), np.array(val)
    raise argparse.ArgumentTypeError(f"Unsupported type for {name}: {type(val).__name__}")


def parse_config_override(spec: str) -> tuple[str, Any]:
    import ast
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--hf-config-override expects key=value, got: {spec!r}")
    key, _, raw = spec.partition("=")
    raw = raw.strip()
    try:
        return key.strip(), ast.literal_eval(raw)
    except Exception:
        return key.strip(), raw


def pool(hidden: np.ndarray, attention_mask: np.ndarray, mode: str) -> np.ndarray:
    """Pool [batch, seq, dim] hidden states to [batch, dim] per `mode`."""
    if mode == "cls":
        return hidden[:, 0, :]
    if mode == "mean":
        m = attention_mask[:, :, None].astype(np.float32)
        return (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    if mode == "last_token":
        idx = attention_mask.sum(axis=1) - 1
        return hidden[np.arange(hidden.shape[0]), idx, :]
    if mode == "pre_pooled":
        if hidden.ndim != 2:
            raise ValueError(f"pre_pooled expects 2D, got shape {hidden.shape}")
        return hidden
    raise ValueError(f"unknown pooling: {mode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", required=True, help="HF repo id of the original PyTorch model")
    p.add_argument("--revision", default=None,
                   help="Pinned commit hash. If omitted, uses HEAD of the local snapshot.")
    p.add_argument("--cache-dir", default="/Volumes/backups/ai/huggingface-hub")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--pooling", required=True,
                   choices=["cls", "mean", "last_token", "pre_pooled"])
    p.add_argument("--text-prefix", default="",
                   help="Prepended to every text before tokenization (rare).")
    p.add_argument("--max-length", type=int, default=512)

    p.add_argument("--threshold", type=float, required=True,
                   help="Min cos_min the Rust test will assert. "
                        "Suggested: 0.999 for FP32, 0.99 for FP16, 0.95 for encoder Q, "
                        "0.90 for decoder INT8/INT4/SmoothQuant.")

    p.add_argument("--extra-input", action="append", default=[], type=parse_extra_input,
                   dest="extra_inputs", metavar="NAME=VALUE",
                   help="Forward to PyTorch: name=value (Python literal). "
                        "For LoRA-adapter models this is informational only.")
    p.add_argument("--hf-config-override", action="append", default=[],
                   type=parse_config_override, dest="hf_config_overrides",
                   metavar="KEY=VALUE",
                   help="Override an HF config attribute when loading the model "
                        "(e.g. --hf-config-override use_memory_efficient_attention=False)")

    p.add_argument("--notes", default="",
                   help="Free-text caveat stored in the fixture metadata "
                        "(e.g. 'V5 Nano probe set excludes one outlier sentence').")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to write the safetensors fixture.")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch, AutoConfig, AutoModel, AutoTokenizer, snapshot_download, save_file = import_deps()

    snap_kwargs = dict(
        repo_id=args.repo,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if args.revision:
        snap_kwargs["revision"] = args.revision
    snap = Path(snapshot_download(**snap_kwargs))
    print(f"HF snapshot: {snap}", flush=True)

    # Resolve the actual commit hash in use (so the fixture pins to a real ref).
    refs_main = (snap.parent.parent / "refs" / "main")
    if args.revision:
        commit = args.revision
    elif refs_main.exists():
        commit = refs_main.read_text().strip()
    else:
        commit = snap.name  # snapshot dir name == hash for local-only

    texts = [args.text_prefix + t for t in TEXTS] if args.text_prefix else list(TEXTS)
    print(f"Tokenizing {len(texts)} texts (max_length={args.max_length}) ...")
    tok = AutoTokenizer.from_pretrained(str(snap), trust_remote_code=args.trust_remote_code)
    enc = tok(texts, padding=True, truncation=True, max_length=args.max_length,
              return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)

    print(f"Loading PyTorch model from {snap} ...")
    config = AutoConfig.from_pretrained(str(snap), trust_remote_code=args.trust_remote_code)
    for k, v in args.hf_config_overrides:
        setattr(config, k, v)
    if args.hf_config_overrides:
        print(f"  config overrides: {dict(args.hf_config_overrides)}")

    model = AutoModel.from_pretrained(
        str(snap),
        config=config,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    with torch.no_grad():
        kwargs = {
            "input_ids": torch.from_numpy(input_ids).long(),
            "attention_mask": torch.from_numpy(attention_mask).long(),
        }
        if "token_type_ids" in enc:
            kwargs["token_type_ids"] = torch.from_numpy(enc["token_type_ids"]).long()
        out = model(**kwargs)
        hidden = out.last_hidden_state.detach().cpu().numpy().astype(np.float32)

    embeddings = pool(hidden, attention_mask, args.pooling).astype(np.float32)
    print(f"Reference embeddings: shape={embeddings.shape} dtype={embeddings.dtype}")
    print(f"  norm range: {np.linalg.norm(embeddings, axis=1).min():.4f}–"
          f"{np.linalg.norm(embeddings, axis=1).max():.4f}")

    metadata = {
        "model_repo": args.repo,
        "revision": commit,
        "pooling": args.pooling,
        "threshold": str(args.threshold),
        "max_length": str(args.max_length),
        "text_prefix": args.text_prefix,
        "texts": json.dumps(texts),
        "notes": args.notes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "embeddings": embeddings,
        },
        str(args.output),
        metadata=metadata,
    )
    size = args.output.stat().st_size
    print(f"\nWrote {args.output} ({size / 1024:.1f} KB)")
    print(f"  pinned to {args.repo}@{commit}")
    print(f"  pooling={args.pooling}  threshold>={args.threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
