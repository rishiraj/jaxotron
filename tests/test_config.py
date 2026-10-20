import pytest

from jaxotron.config import MODEL_PRESETS, Config, ModelConfig, ParallelConfig, TrainConfig, apply_override


def test_presets_are_valid_single_device():
    for model in MODEL_PRESETS.values():
        Config(model=model).validate(1)  # raises on inconsistent shapes


@pytest.mark.parametrize("parallel, message", [
    (dict(tp=3), "num_key_value_heads"),
    (dict(pp=3), "num_hidden_layers"),
    (dict(cp=3), "sequence_length"),
    (dict(fsdp=3), "hidden_size"),
])
def test_invalid_layouts_have_clear_errors(parallel, message):
    cfg = Config(parallel=ParallelConfig(**parallel))
    with pytest.raises(ValueError, match=message):
        cfg.validate(ParallelConfig(**parallel).world_size)


def test_device_count_mismatch_suggests_cpu_devices():
    with pytest.raises(ValueError, match="--cpu_devices 4"):
        Config(parallel=ParallelConfig(dp=2, tp=2)).validate(1)


def test_overrides_and_roundtrip():
    cfg = Config()
    apply_override(cfg, "train.learning_rate", "3e-4")
    apply_override(cfg, "model.rope_theta", "5e5")
    assert cfg.train.learning_rate == 3e-4 and cfg.model.rope_theta == 5e5
    with pytest.raises(AttributeError):
        apply_override(cfg, "train.does_not_exist", "1")
    d = cfg.to_dict()
    d["model"]["field_from_the_future"] = 1  # unknown keys are ignored
    d["removed_top_level_option"] = True
    assert Config.from_dict(d).train == cfg.train


def test_grad_accum_and_pipeline_microbatches():
    cfg = Config(parallel=ParallelConfig(dp=2, pp=2),
                 train=TrainConfig(global_batch_size=32, micro_batch_size=4))
    assert cfg.grad_accum_steps == 4 and cfg.pp_microbatches == 4
    assert ModelConfig(hidden_size=64, num_attention_heads=4).head_dim == 16
