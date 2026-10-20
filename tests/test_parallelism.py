"""Every parallel layout must compute the same loss and gradients as one device."""
import jax
import numpy as np
import pytest

from jaxotron.config import Config, ModelConfig, ParallelConfig, TrainConfig
from jaxotron.data.loader import to_global_array
from jaxotron.parallel.mesh import create_mesh
from jaxotron.parallel.rules import batch_sharding
from jaxotron.trainer import init_model, make_context, make_grad_fn

MODEL = ModelConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2)
SEQ, GBS = 16, 16

LAYOUTS = {
    "single":      dict(),
    "dp8":         dict(dp=8),
    "fsdp8":       dict(fsdp=8),
    "tp2":         dict(tp=2),
    "pp4":         dict(pp=4),
    "cp4":         dict(cp=4),
    "dp2_fsdp2_tp2": dict(dp=2, fsdp=2, tp=2),
    "4d_dp_pp_cp_tp": dict(dp=2, pp=2, cp=2, tp=2),
    "4d_fsdp_pp_cp_tp": dict(fsdp=2, pp=2, cp=2, tp=2),
}


def _run(layout: dict, accum: int = 1, remat: str = "full"):
    pcfg = ParallelConfig(**layout)
    dp = pcfg.data_parallel_size
    cfg = Config(parallel=pcfg, model=MODEL,
                 train=TrainConfig(global_batch_size=GBS, micro_batch_size=GBS // (dp * accum),
                                   sequence_length=SEQ, dtype="float32", remat=remat))
    devices = jax.devices()[: pcfg.world_size]
    cfg.validate(len(devices))
    mesh = create_mesh(pcfg, devices)
    ctx = make_context(cfg, mesh)
    params, static, _ = init_model(MODEL, mesh, jax.random.PRNGKey(0))
    tokens = np.random.default_rng(0).integers(0, MODEL.vocab_size, (GBS, SEQ + 1)).astype(np.int32)
    batch = to_global_array(tokens, batch_sharding(mesh))
    loss, grads = jax.jit(make_grad_fn(static, ctx, cfg.grad_accum_steps))(params, batch)
    return float(loss), jax.tree.map(np.asarray, grads)


@pytest.fixture(scope="module")
def reference():
    return _run({})


def _assert_matches(result, reference):
    (loss, grads), (ref_loss, ref_grads) = result, reference
    assert np.isclose(loss, ref_loss, rtol=1e-5), (loss, ref_loss)
    for g, r in zip(jax.tree.leaves(grads), jax.tree.leaves(ref_grads)):
        np.testing.assert_allclose(g, r, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("name", [n for n in LAYOUTS if n != "single"])
def test_layout_matches_single_device(name, reference):
    _assert_matches(_run(LAYOUTS[name]), reference)


def test_gradient_accumulation_matches(reference):
    _assert_matches(_run(dict(dp=2, pp=2), accum=2), reference)


@pytest.mark.parametrize("remat", ["dots", "none"])
def test_remat_policies_match(remat, reference):
    _assert_matches(_run(dict(tp=2, cp=2), remat=remat), reference)
