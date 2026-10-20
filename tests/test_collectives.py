"""The collectives XLA inserts match the strategy table in the README."""
import jax
import numpy as np
import pytest

from jaxotron.config import Config, ModelConfig, ParallelConfig, TrainConfig
from jaxotron.data.loader import to_global_array
from jaxotron.parallel.collectives import _parse_groups, collectives_table, count_collectives
from jaxotron.parallel.mesh import create_mesh
from jaxotron.parallel.rules import batch_sharding
from jaxotron.trainer import init_model, make_context, make_grad_fn

MODEL = ModelConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2)


def _collectives(**layout):
    pcfg = ParallelConfig(**layout)
    cfg = Config(parallel=pcfg, model=MODEL,
                 train=TrainConfig(global_batch_size=8, micro_batch_size=8 // pcfg.data_parallel_size,
                                   sequence_length=16, dtype="float32"))
    mesh = create_mesh(pcfg, jax.devices()[:pcfg.world_size])
    params, static, _ = init_model(MODEL, mesh, jax.random.PRNGKey(0))
    batch = to_global_array(np.zeros((8, 17), np.int32), batch_sharding(mesh))
    hlo = jax.jit(make_grad_fn(static, make_context(cfg, mesh), 1)).lower(params, batch).compile().as_text()
    stats = count_collectives(hlo, mesh)
    print(collectives_table(stats))
    return {(op, axes) for op, axes in stats}


def _axes_used(found, op=None):
    return {a for o, axes in found if op is None or o == op for a in axes}


def test_iota_and_list_replica_groups():
    assert _parse_groups("replica_groups={{0,1},{2,3}}, to_apply") == [[0, 1], [2, 3]]
    assert _parse_groups("replica_groups=[2,2]<=[4]") == [[0, 1], [2, 3]]
    assert _parse_groups("replica_groups=[2,2]<=[2,2]T(1,0)") == [[0, 2], [1, 3]]
    assert _parse_groups("source_target_pairs={{0,1},{1,0}}") == [[0, 1], [1, 0]]


@pytest.mark.parametrize("layout, axis", [(dict(dp=2), "dp"), (dict(tp=2), "tp"), (dict(fsdp=2), "fsdp"),
                                          (dict(cp=2), "cp"), (dict(pp=2), "pp")])
def test_each_strategy_communicates_over_its_axis(layout, axis):
    assert _axes_used(_collectives(**layout)) == {axis}


def test_pipeline_uses_collective_permute():
    assert "pp" in _axes_used(_collectives(pp=2), "collective-permute")


def test_single_device_has_no_collectives():
    assert _collectives() == set()


def test_mesh_format_replica_groups():
    line = "all-reduce(x), replica_groups=mesh['dp'=2,'tp'=2] {'tp'}, use_global_device_ids=true"
    assert _parse_groups(line) == [[0, 1], [2, 3]]
    line = "all-reduce(x), replica_groups=mesh['dp'=2,'tp'=2] {'dp'}, to_apply=%r"
    assert _parse_groups(line) == [[0, 2], [1, 3]]
