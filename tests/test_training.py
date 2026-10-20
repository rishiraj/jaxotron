"""Short end-to-end runs: learning happens, and checkpoints move between meshes."""
import dataclasses

import jax
import numpy as np
import pytest

from jaxotron.checkpoint import Checkpointer
from jaxotron.config import Config, ModelConfig, ParallelConfig, TrainConfig
from jaxotron.data.synthetic import write_synthetic_dataset
from jaxotron.parallel.mesh import create_mesh
from jaxotron.trainer import abstract_model, init_model, train

MODEL = ModelConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2)


def _config(tmp_path, steps, **parallel):
    write_synthetic_dataset(str(tmp_path / "data"), train_tokens=50_000, val_tokens=5_000, vocab_size=128)
    pcfg = ParallelConfig(**parallel)
    return Config(
        parallel=pcfg, model=MODEL,
        train=TrainConfig(global_batch_size=16, micro_batch_size=16 // (2 * pcfg.data_parallel_size),
                          sequence_length=32, train_steps=steps, learning_rate=3e-3, warmup_steps=5,
                          log_every=10, save_every=steps, dtype="float32", eval_every=steps, eval_batches=2),
        data_path=str(tmp_path / "data" / "train.bin"), val_path=str(tmp_path / "data" / "val.bin"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )


def _use_devices(monkeypatch, n):
    devices = jax.devices()[:n]
    monkeypatch.setattr(jax, "device_count", lambda: n)
    monkeypatch.setattr(jax, "devices", lambda *a: devices)


@pytest.mark.slow
def test_4d_training_learns_and_resumes_on_another_mesh(tmp_path, monkeypatch):
    _use_devices(monkeypatch, 16)
    cfg = _config(tmp_path, steps=40, dp=2, pp=2, cp=2, tp=2)
    cfg.show_collectives = cfg.show_mesh = True
    first = train(cfg)
    assert first["train/loss"] < 3.5, first  # ln(128) = 4.85 at initialisation
    assert "val/loss" in first

    # Restore the 4D checkpoint on a different 2-device layout and keep training.
    _use_devices(monkeypatch, 2)
    cfg2 = dataclasses.replace(_config(tmp_path, steps=50, fsdp=2), checkpoint_dir=cfg.checkpoint_dir)
    second = train(cfg2)
    assert second["train/loss"] < first["train/loss"] + 0.5


def test_dry_run_compiles(tmp_path, monkeypatch, capsys):
    _use_devices(monkeypatch, 8)
    cfg = _config(tmp_path, steps=5, dp=2, pp=2, tp=2)
    cfg.dry_run = cfg.show_collectives = True
    assert train(cfg) == {}
    out = capsys.readouterr().out
    assert "collective-permute" in out and "Dry run complete" in out


def test_checkpoint_params_identical_across_meshes(tmp_path):
    mesh_a = create_mesh(ParallelConfig(pp=2, tp=2, fsdp=2), jax.devices()[:8])
    params, _, _ = init_model(MODEL, mesh_a, jax.random.PRNGKey(1))
    ckpt = Checkpointer(str(tmp_path / "c"))
    ckpt.save(1, params, {"dummy": params.final_norm})
    ckpt.wait()
    assert not ckpt.save(1, params, {"dummy": params.final_norm})  # same step twice is a no-op

    mesh_b = create_mesh(ParallelConfig(), jax.devices()[:1])
    abstract, _, _ = abstract_model(MODEL, mesh_b)
    _, restored = ckpt.restore_params(abstract)
    ckpt.close()
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert len(b.sharding.device_set) == 1