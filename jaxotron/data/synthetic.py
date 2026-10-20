"""A learnable synthetic dataset for zero-download demos.

Documents are arithmetic progressions modulo the vocabulary, e.g.
    17, 20, 23, 26, ...   (start=17, stride=3)
After seeing two tokens the rest is predictable, so the loss visibly drops
from ~ln(vocab) towards zero within a few hundred steps. Perfect for live demos
on a laptop without internet.
"""
from __future__ import annotations

import json
import os

import numpy as np

EOS = 0


def make_synthetic_tokens(num_tokens: int, vocab_size: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(num_tokens, dtype=np.uint16)
    pos = 0
    while pos < num_tokens:
        length = int(rng.integers(32, 256))
        start, stride = int(rng.integers(0, vocab_size - 1)), int(rng.integers(1, 8))
        doc = 1 + (start + stride * np.arange(length)) % (vocab_size - 1)  # ids 1..V-1, 0 is EOS
        chunk = np.append(doc, EOS)
        n = min(len(chunk), num_tokens - pos)
        out[pos:pos + n] = chunk[:n]
        pos += n
    return out


def write_synthetic_dataset(out_dir: str, train_tokens: int = 2_000_000, val_tokens: int = 100_000,
                            vocab_size: int = 1024, seed: int = 0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, n, s in (("train", train_tokens, seed), ("val", val_tokens, seed + 1)):
        path = os.path.join(out_dir, f"{name}.bin")
        tmp = f"{path}.tmp.{os.getpid()}"
        make_synthetic_tokens(n, vocab_size, s).tofile(tmp)
        os.replace(tmp, path)  # atomic: safe if several hosts generate concurrently
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"vocab_size": vocab_size, "dtype": "uint16", "tokenizer": None,
                   "description": "synthetic arithmetic progressions"}, f, indent=2)
