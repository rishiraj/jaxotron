"""Jaxotron training entry point (`jaxotron-train` or `python train.py`).

Examples
--------
# 4D parallelism (DP x PP x CP x TP) on a laptop, using 16 simulated CPU devices:
jaxotron-train --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2

# What communication did XLA insert for this layout?
jaxotron-train --cpu_devices 8 --dp 2 --pp 2 --tp 2 --dry_run --show_collectives

# Continue pre-training SmolLM2-135M on TinyStories with FSDP on 8 GPUs / a TPU v5e-8:
jaxotron-train --init_from_hf HuggingFaceTB/SmolLM2-135M --fsdp 8 --data data/tinystories/train.bin \
    --seq_len 1024 --global_batch_size 64 --micro_batch_size 8 --lr 3e-4 --steps 1000

# Any config field can be overridden:
jaxotron-train --set train.weight_decay=0.05 --set model.rope_theta=500000
"""
from __future__ import annotations

import argparse
import os


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Jaxotron: 4D-parallel LLM pre-training in JAX",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("parallelism (product must equal the number of devices)")
    for axis, desc in (("dp", "data"), ("fsdp", "fully-sharded data (ZeRO-3)"), ("pp", "pipeline"),
                       ("cp", "context (sequence)"), ("tp", "tensor")):
        g.add_argument(f"--{axis}", type=int, default=1, help=f"{desc} parallel size")
    g.add_argument("--pp_microbatches", type=int, default=0, help="pipeline microbatches (0 = micro_batch_size)")

    g = ap.add_argument_group("model")
    g.add_argument("--preset", default="tiny", help="architecture when training from scratch (see --list_presets)")
    g.add_argument("--list_presets", action="store_true", help="print the model presets and exit")
    g.add_argument("--init_from_hf", default=None, metavar="REPO_OR_DIR",
                   help="start from pretrained Llama-family weights, e.g. HuggingFaceTB/SmolLM2-135M")
    g.add_argument("--vocab_size", type=int, default=None, help="default: inferred from the dataset's meta.json")

    g = ap.add_argument_group("training")
    g.add_argument("--seq_len", type=int, default=256)
    g.add_argument("--global_batch_size", type=int, default=32)
    g.add_argument("--micro_batch_size", type=int, default=4, help="per data-parallel replica")
    g.add_argument("--steps", type=int, default=300)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--warmup", type=int, default=30)
    g.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"])
    g.add_argument("--remat", default="full", choices=["full", "dots", "none"])
    g.add_argument("--attention", default="reference", choices=["reference", "cudnn"])
    g.add_argument("--log_every", type=int, default=10)
    g.add_argument("--eval_every", type=int, default=0)
    g.add_argument("--save_every", type=int, default=500)
    g.add_argument("--seed", type=int, default=42)

    g = ap.add_argument_group("io")
    g.add_argument("--data", default="data/synthetic/train.bin")
    g.add_argument("--val_data", default=None, help="default: val.bin next to --data, if present")
    g.add_argument("--checkpoint_dir", default="checkpoints/run")
    g.add_argument("--run_name", default="jaxotron")
    g.add_argument("--wandb_project", default=None)

    g = ap.add_argument_group("runtime and inspection")
    g.add_argument("--cpu_devices", type=int, default=0, help="simulate N devices on CPU (great for demos)")
    g.add_argument("--distributed", action="store_true", help="multi-host: call jax.distributed.initialize()")
    g.add_argument("--dry_run", action="store_true", help="compile one step, print memory, exit")
    g.add_argument("--show_sharding", action="store_true", help="print how every parameter is sharded")
    g.add_argument("--show_mesh", action="store_true", help="print the mesh coordinates of every device")
    g.add_argument("--show_collectives", action="store_true",
                   help="print the collectives XLA inserted, grouped by mesh axis")
    g.add_argument("--profile_dir", default=None, help="write a profiler trace of a few steps here")
    g.add_argument("--peak_tflops", type=float, default=None,
                   help="per-device peak bf16 TFLOP/s for MFU when the hardware is not auto-detected")
    g.add_argument("--compile_cache", default="~/.cache/jaxotron/xla",
                   help="persistent XLA compilation cache ('' disables)")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config field, e.g. train.weight_decay=0.05")
    return ap.parse_args(argv)


def setup_cpu_devices(n: int) -> None:
    """Simulate n devices on the host CPU. Must run before JAX is imported."""
    if n:
        os.environ["JAX_PLATFORMS"] = "cpu"
        flags = [f for f in os.environ.get("XLA_FLAGS", "").split() if "xla_force_host_platform_device_count" not in f]
        os.environ["XLA_FLAGS"] = " ".join(flags + [f"--xla_force_host_platform_device_count={n}"])


def main(argv=None):
    args = parse_args(argv)
    setup_cpu_devices(args.cpu_devices)

    import dataclasses

    from jaxotron.config import MODEL_PRESETS, Config, ParallelConfig, TrainConfig, apply_override
    from jaxotron.data.loader import read_meta

    if args.list_presets:
        from jaxotron.models.llama import count_params
        for name, m in MODEL_PRESETS.items():
            print(f"{name:<14} {count_params(m) / 1e6:8.1f}M params | layers={m.num_hidden_layers} "
                  f"hidden={m.hidden_size} heads={m.num_attention_heads}/{m.num_key_value_heads} vocab={m.vocab_size}")
        return
    if args.preset not in MODEL_PRESETS:
        raise SystemExit(f"Unknown preset {args.preset!r}. Choose from {sorted(MODEL_PRESETS)}")

    # Zero-setup demo data.
    if not os.path.exists(args.data) and os.path.normpath(args.data) == os.path.normpath("data/synthetic/train.bin"):
        from jaxotron.data.synthetic import write_synthetic_dataset
        print("Generating synthetic dataset in data/synthetic ...")
        write_synthetic_dataset("data/synthetic")

    meta_vocab = read_meta(args.data).get("vocab_size")
    if args.init_from_hf:
        from jaxotron.hf import hf_model_config, resolve_hf_path
        args.init_from_hf = resolve_hf_path(args.init_from_hf)
        model = hf_model_config(args.init_from_hf)
    else:
        model = MODEL_PRESETS[args.preset]
        if args.vocab_size:
            model = dataclasses.replace(model, vocab_size=args.vocab_size)
        elif meta_vocab:
            padded = -(-meta_vocab // 128) * 128  # pad to a multiple of 128 for efficient sharding
            if padded != model.vocab_size:
                print(f"vocab_size inferred from dataset: {meta_vocab} -> {padded} (padded)")
            model = dataclasses.replace(model, vocab_size=padded)

    val_path = args.val_data
    if val_path is None:
        candidate = os.path.join(os.path.dirname(args.data), "val.bin")
        val_path = candidate if os.path.exists(candidate) else None

    cfg = Config(
        parallel=ParallelConfig(dp=args.dp, fsdp=args.fsdp, pp=args.pp, cp=args.cp, tp=args.tp,
                                pp_microbatches=args.pp_microbatches),
        model=model,
        train=TrainConfig(global_batch_size=args.global_batch_size, micro_batch_size=args.micro_batch_size,
                          sequence_length=args.seq_len, train_steps=args.steps, learning_rate=args.lr,
                          warmup_steps=args.warmup, dtype=args.dtype, remat=args.remat,
                          attention_impl=args.attention, log_every=args.log_every,
                          eval_every=args.eval_every, save_every=args.save_every, seed=args.seed),
        data_path=args.data, val_path=val_path, checkpoint_dir=args.checkpoint_dir,
        init_from_hf=args.init_from_hf, run_name=args.run_name, wandb_project=args.wandb_project,
        distributed=args.distributed, profile_dir=args.profile_dir, dry_run=args.dry_run,
        show_sharding=args.show_sharding, show_mesh=args.show_mesh, show_collectives=args.show_collectives,
        peak_tflops=args.peak_tflops,
    )
    for item in args.set:
        key, _, value = item.partition("=")
        apply_override(cfg, key.strip(), value.strip())

    from jaxotron.utils import enable_compilation_cache
    enable_compilation_cache(args.compile_cache)

    from jaxotron.trainer import train
    try:
        train(cfg)
    except ValueError as e:
        if str(e).startswith("Invalid configuration"):
            raise SystemExit(str(e)) from None
        raise


if __name__ == "__main__":
    main()
