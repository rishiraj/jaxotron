import numpy as np

from jaxotron.data.loader import StridePermutation, TokenDataset, batch_indices
from jaxotron.data.synthetic import write_synthetic_dataset


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
