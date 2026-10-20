import jax
import numpy as np

from jaxotron.config import ParallelConfig
from jaxotron.data.loader import DataLoader, StridePermutation, TokenDataset, batch_indices, local_rows
from jaxotron.data.synthetic import write_synthetic_dataset
from jaxotron.parallel.mesh import create_mesh
from jaxotron.parallel.rules import batch_sharding


def test_stride_permutation_is_a_permutation():
    for n in (1, 2, 7, 64, 1000):
        perm = StridePermutation(n, seed=3)
        out = np.asarray(perm(0, np.arange(n)), dtype=np.int64)
        assert sorted(out.tolist()) == list(range(n))


def test_batches_are_deterministic_and_resumable():
    perm_a, perm_b = StridePermutation(100, 0), StridePermutation(100, 0)
    full = [batch_indices(s, 16, 100, perm_a) for s in range(20)]
    resumed = [batch_indices(s, 16, 100, perm_b) for s in range(10, 20)]
    for a, b in zip(full[10:], resumed):
        np.testing.assert_array_equal(a, b)


def test_epoch_covers_every_sample():
    perm = StridePermutation(64, 1)
    seen = np.concatenate([batch_indices(s, 8, 64, perm) for s in range(8)])
    assert sorted(seen.tolist()) == list(range(64))


def test_dataset_samples_are_shifted_windows(tmp_path):
    write_synthetic_dataset(str(tmp_path), train_tokens=10_000, val_tokens=1_000)
    ds = TokenDataset(str(tmp_path / "train.bin"), seq_len=32)
    assert ds[0].shape == (33,) and ds[0].dtype == np.int32
    assert ds[1][0] == ds[0][32]  # consecutive samples overlap by one token (inputs/labels shift)


def test_loader_produces_sharded_global_batches(tmp_path):
    write_synthetic_dataset(str(tmp_path), train_tokens=10_000, val_tokens=1_000)
    ds = TokenDataset(str(tmp_path / "train.bin"), seq_len=16)
    mesh = create_mesh(ParallelConfig(dp=2, fsdp=2), jax.devices()[:4])
    sharding = batch_sharding(mesh)
    assert local_rows(8, sharding, 17).tolist() == list(range(8))  # single host owns every row
    with DataLoader(ds, 8, sharding, seed=0, start_step=3) as loader:
        batch = next(loader)
        expected = np.stack([ds[int(i)] for i in batch_indices(3, 8, len(ds), loader.perm)])
    assert batch.shape == (8, 17) and batch.sharding == sharding
    np.testing.assert_array_equal(np.asarray(batch), expected)