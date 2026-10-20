"""Pure NumPy + JAX data pipeline (no PyTorch dependency).

* TokenDataset memory-maps a flat binary file of token ids, so datasets far
  larger than RAM open instantly.
* The sample order is a pure function of (seed, step): resuming from a
  checkpoint resumes the exact same data stream, on any number of hosts.
* Batches become *global* jax.Arrays with jax.make_array_from_callback. Each
  host reads from disk only the rows that its own devices need, which keeps
  the loader cheap on multi-host GPU clusters and TPU pods.
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading
from typing import Optional

import jax
import numpy as np
from jax.sharding import NamedSharding


def read_meta(data_path: str) -> dict:
    meta_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


class TokenDataset:
    """Fixed-length samples of seq_len + 1 tokens (inputs and shifted labels)."""

    def __init__(self, path: str, seq_len: int, dtype: Optional[str] = None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}. Run `python prepare_data.py --help`.")
        dtype = dtype or read_meta(path).get("dtype", "uint16")
        self.tokens = np.memmap(path, dtype=np.dtype(dtype), mode="r")
        self.seq_len = seq_len
        self.num_samples = (len(self.tokens) - 1) // seq_len
        if self.num_samples < 1:
            raise ValueError(f"{path} has {len(self.tokens)} tokens, fewer than seq_len + 1 = {seq_len + 1}.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int) -> np.ndarray:
        start = i * self.seq_len
        return np.asarray(self.tokens[start:start + self.seq_len + 1], dtype=np.int32)


class StridePermutation:
    """O(1)-memory pseudo-random permutation of range(n): i -> (a*i + b) mod n, with gcd(a, n) = 1.

    A new (a, b) is drawn per epoch. Works for billions of samples without
    materialising an index array.
    """

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self._cache = n, seed, {}

    def _params(self, epoch: int) -> tuple[int, int]:
        if epoch not in self._cache:
            rng = np.random.default_rng((self.seed, epoch))
            a = 1
            if self.n > 2:
                a = int(rng.integers(1, self.n))
                while math.gcd(a, self.n) != 1:
                    a = int(rng.integers(1, self.n))
            self._cache = {epoch: (a, int(rng.integers(0, self.n)))}
        return self._cache[epoch]

    def __call__(self, epoch: int, positions: np.ndarray) -> np.ndarray:
        a, b = self._params(epoch)
        return (a * positions.astype(object) + b) % self.n  # object dtype avoids int64 overflow


def batch_indices(step: int, batch_size: int, num_samples: int, perm: Optional[StridePermutation]) -> np.ndarray:
    global_ids = np.arange(step * batch_size, (step + 1) * batch_size, dtype=np.int64)
    epochs, positions = global_ids // num_samples, global_ids % num_samples
    if perm is None:
        return positions
    out = np.empty_like(positions)
    for epoch in np.unique(epochs):
        mask = epochs == epoch
        out[mask] = np.asarray(perm(int(epoch), positions[mask]), dtype=np.int64)
    return out


def local_rows(batch_size: int, sharding: NamedSharding, row_len: int) -> np.ndarray:
    """Global batch rows owned by this host's devices (all rows on a single host)."""
    if jax.process_count() == 1:
        return np.arange(batch_size)
    rows = set()
    for idx in sharding.addressable_devices_indices_map((batch_size, row_len)).values():
        rows.update(range(*idx[0].indices(batch_size)))
    return np.array(sorted(rows), dtype=np.int64)


def to_global_array(batch: np.ndarray, sharding: NamedSharding) -> jax.Array:
    """Build a global jax.Array; each host only copies the shards its devices own."""
    return jax.make_array_from_callback(batch.shape, sharding, lambda idx: batch[idx])


class DataLoader:
    """Infinite, deterministic, resumable iterator of global token batches with background prefetch."""

    def __init__(self, dataset: TokenDataset, batch_size: int, sharding: NamedSharding, *,
                 seed: int = 0, start_step: int = 0, shuffle: bool = True, prefetch: int = 2):
        self.dataset, self.batch_size, self.sharding = dataset, batch_size, sharding
        self.perm = StridePermutation(len(dataset), seed) if shuffle else None
        self.rows = local_rows(batch_size, sharding, dataset.seq_len + 1)
        self.step = start_step
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, args=(start_step,), daemon=True)
        self._thread.start()

    def host_batch(self, step: int) -> np.ndarray:
        """[batch_size, seq_len + 1] int32. Rows owned by other hosts are left as zeros (never read)."""
        idx = batch_indices(step, self.batch_size, len(self.dataset), self.perm)
        batch = np.zeros((self.batch_size, self.dataset.seq_len + 1), np.int32)
        for r in self.rows:
            batch[r] = self.dataset[int(idx[r])]
        return batch

    def _worker(self, step: int) -> None:
        while not self._stop.is_set():
            try:
                item = self.host_batch(step)
            except Exception as e:  # surface errors in the main thread
                item = e
            while not self._stop.is_set():
                try:
                    self._queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if isinstance(item, Exception):
                return
            step += 1

    def __iter__(self):
        return self

    def __next__(self) -> jax.Array:
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        self.step += 1
        return to_global_array(item, self.sharding)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
