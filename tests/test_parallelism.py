"""Every parallel layout must compute the same loss and gradients as one device."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxotron.config import Config, ModelConfig, ParallelConfig, TrainConfig
from jaxotron.data.loader import to_global_array
from jaxotron.models.llama import forward, forward_cached, init_kv_cache
from jaxotron.parallel.mesh import create_mesh
from jaxotron.parallel.rules import batch_sharding
from jaxotron.trainer import init_model, make_context, make_grad_fn

MODEL = ModelConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2)
SEQ, GBS = 16, 16

LAYOUTS = {
    "single":             dict(),
    "dp8":                dict(dp=8),
    "fsdp8":              dict(fsdp=8),
    "tp2":                dict(tp=2),
    "tp2_sp":             dict(tp=2, sp=True),
    "pp4":                dict(pp=4),
    "cp4":                dict(cp=4),
    "dp2_fsdp2_tp2":      dict(dp=2, fsdp=2, tp=2),
    "4d_dp_pp_cp_tp":     dict(dp=2, pp=2, cp=2, tp=2),
    "4d_dp_pp_cp_tp_sp":  dict(dp=2, pp=2, cp=2, tp=2, sp=True),
    "4d_fsdp_pp_cp_tp":   dict(fsdp=2, pp=2, cp=2, tp=2),
}


def _run(layout: dict, accum: int = 1, remat: str = "full", model: ModelConfig = MODEL):
    pcfg = ParallelConfig(**layout)
    dp = pcfg.data_parallel_size
    cfg = Config(parallel=pcfg, model=model,
                 train=TrainConfig(global_batch_size=GBS, micro_batch_size=GBS // (dp * accum),
                                   sequence_length=SEQ, dtype="float32", remat=remat))
    devices = jax.devices()[: pcfg.world_size]
    cfg.validate(len(devices))
    mesh = create_mesh(pcfg, devices)
    ctx = make_context(cfg, mesh)
    params, static, _ = init_model(model, mesh, jax.random.PRNGKey(0))
    tokens = np.random.default_rng(0).integers(0, model.vocab_size, (GBS, SEQ + 1)).astype(np.int32)
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


def test_qwen3_style_4d_matches_single_device():
    qwen_model = ModelConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                             num_attention_heads=4, num_key_value_heads=2, attn_head_dim=32, qk_norm=True)
    ref = _run({}, model=qwen_model)
    res = _run(dict(dp=2, pp=2, cp=2, tp=2), model=qwen_model)
    _assert_matches(res, ref)


def test_gradient_accumulation_matches(reference):
    _assert_matches(_run(dict(dp=2, pp=2), accum=2), reference)


@pytest.mark.parametrize("remat", ["dots", "none"])
def test_remat_policies_match(remat, reference):
    _assert_matches(_run(dict(tp=2, cp=2), remat=remat), reference)


@pytest.mark.parametrize("tp", [1, 2])
def test_kv_cached_decoding_matches_full_forward(tp):
    pcfg = ParallelConfig(tp=tp)
    mesh = create_mesh(pcfg, jax.devices()[:tp])
    cfg = Config(parallel=pcfg, model=MODEL, train=TrainConfig(dtype="float32", remat="none"))
    ctx = make_context(cfg, mesh)
    params, static, _ = init_model(MODEL, mesh, jax.random.PRNGKey(0))
    model = eqx.combine(params, static)

    seq = jnp.asarray([[3, 7, 11, 15, 19, 23, 27, 31]], dtype=jnp.int32)
    full_logits = np.asarray(jax.jit(lambda m, s: forward(m, s, ctx))(model, seq))

    cache = init_kv_cache(MODEL, batch_size=1, max_seq=8, ctx=ctx)
    prefill_logits, cache = jax.jit(lambda m, s, c: forward_cached(m, s, jnp.int32(0), c, ctx))(model, seq[:, :4], cache)
    np.testing.assert_allclose(np.asarray(prefill_logits), full_logits[:, :4], rtol=1e-5, atol=1e-5)

    for pos in range(4, 8):
        step_logits, cache = jax.jit(lambda m, t, p, c: forward_cached(m, t, p, c, ctx))(
            model, seq[:, pos:pos + 1], jnp.int32(pos), cache
        )
        np.testing.assert_allclose(np.asarray(step_logits[:, 0]), full_logits[:, pos], rtol=1e-5, atol=1e-5)