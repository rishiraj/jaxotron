"""Tokenise a Hugging Face dataset into flat binary token files for Jaxotron.

Output: <out_dir>/train.bin, <out_dir>/val.bin, <out_dir>/meta.json

# Real text (needs: pip install "jaxotron[data]")
jaxotron-prepare-data --dataset roneneldan/TinyStories --tokenizer HuggingFaceTB/SmolLM2-135M \
    --out_dir data/tinystories --max_docs 200000

# Synthetic, learnable data (no downloads, no extra dependencies)
jaxotron-prepare-data --synthetic --out_dir data/synthetic
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def tokenize(args) -> None:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit('Tokenising real data needs: pip install "jaxotron[data]"') from None

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    dtype = np.uint16 if vocab_size <= 2**16 else np.uint32
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    ds = load_dataset(args.dataset, name=args.subset, split=args.split, streaming=True)
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {s: os.path.join(args.out_dir, f"{s}.bin") for s in ("train", "val")}
    tmp_paths = {s: f"{p}.tmp" for s, p in paths.items()}
    files = {s: open(p, "wb") for s, p in tmp_paths.items()}
    counts = {"train": 0, "val": 0}
    val_every = max(int(1 / args.val_fraction), 1) if args.val_fraction > 0 else 0

    def flush(texts, doc_offset):
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for j, doc in enumerate(ids):
            split = "val" if val_every and (doc_offset + j) % val_every == 0 else "train"
            arr = np.asarray(doc + [eos], dtype=dtype)
            arr.tofile(files[split])
            counts[split] += len(arr)

    batch, n_docs = [], 0
    try:
        for example in ds:
            text = example[args.text_column]
            if not text:
                continue
            batch.append(text)
            n_docs += 1
            if len(batch) == 1000:
                flush(batch, n_docs - len(batch))
                batch = []
                print(f"\r{n_docs:,} docs | {counts['train']:,} train tokens", end="", flush=True)
            if args.max_docs and n_docs >= args.max_docs:
                break
        if batch:
            flush(batch, n_docs - len(batch))
    finally:
        for f in files.values():
            f.close()
    for s in paths:  # only publish complete files
        os.replace(tmp_paths[s], paths[s])

    meta = {"vocab_size": vocab_size, "dtype": np.dtype(dtype).name, "tokenizer": args.tokenizer,
            "dataset": args.dataset, "docs": n_docs, "tokens": counts}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {counts['train']:,} train / {counts['val']:,} val tokens to {args.out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_column", default="text")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--out_dir", default="data/tinystories")
    ap.add_argument("--max_docs", type=int, default=0, help="0 = whole dataset")
    ap.add_argument("--val_fraction", type=float, default=0.005)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic_tokens", type=int, default=2_000_000)
    args = ap.parse_args(argv)

    if args.synthetic:
        from jaxotron.data.synthetic import write_synthetic_dataset
        write_synthetic_dataset(args.out_dir, train_tokens=args.synthetic_tokens)
        print(f"Wrote synthetic dataset to {args.out_dir}")
    else:
        tokenize(args)


if __name__ == "__main__":
    main()
