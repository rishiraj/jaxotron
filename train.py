"""Jaxotron training entry point.

Examples
--------
# 4D parallelism (DP x PP x CP x TP) on a laptop, using 16 simulated CPU devices:
python train.py --cpu_devices 16 --dp 2 --pp 2 --cp 2 --tp 2

# 8 GPUs, ZeRO-3 style:
python train.py --preset 125m --fsdp 8 --data data/tinystories/train.bin --seq_len 1024 \
    --global_batch_size 256 --micro_batch_size 8 --lr 6e-4 --steps 5000

# Any config field can be overridden:
python train.py --set train.weight_decay=0.05 --set model.rope_theta=500000
"""
from __future__ import annotations

import argparse
import os


def parse_args():
    ap = argparse.ArgumentParser(description="Jaxotron: 4D-parallel LLM pre-training in JAX",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("parallelism (product must equal the number of devices)")
    for axis, desc in (("dp", "data"), ("fsdp", "fully-sharded data (ZeRO-3)"), ("pp", "pipeline"),
                       ("cp", "context (sequence)"), ("tp", "tensor")):
        g.add_argument(f"--{axis}", type=int, default=1, help=f"{desc} parallel size")
    g.add_argument("--pp_microbatches", type=int, default=0, help="pipeline microbatches (0 = auto)")

    g = ap.add_argument_group("model")
    g.add_argument("--preset", default="tiny", help="tiny | 125m | smollm2-1.7b | llama3.2-1b | llama3.1-8b")
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

    g = ap.add_argument_group("runtime")
    g.add_argument("--cpu_devices", type=int, default=0, help="simulate N devices on CPU (great for demos)")
    g.add_argument("--distributed", action="store_true", help="multi-host: call jax.distributed.initialize()")
    g.add_argument("--dry_run", action="store_true", help="compile one step, print memory, exit")
    g.add_argument("--show_sharding", action="store_true", help="print how every parameter is sharded")
    g.add_argument("--profile_dir", default=None, help="write a profiler trace of a few steps here")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config field, e.g. train.weight_decay=0.05")
    return ap.parse_args()


def main():
    args = parse_args()

    # Must happen before JAX is imported anywhere.
    if args.cpu_devices:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   f" --xla_force_host_platform_device_count={args.cpu_devices}")

    import dataclasses
    from jaxotron.config import MODEL_PRESETS, Config, ParallelConfig, TrainConfig, apply_override
    from jaxotron.data.loader import read_meta

    if args.preset not in MODEL_PRESETS:
        raise SystemExit(f"Unknown preset {args.preset!r}. Choose from {sorted(MODEL_PRESETS)}")

    # Zero-setup demo data.
    if not os.path.exists(args.data) and os.path.normpath(args.data) == os.path.normpath("data/synthetic/train.bin"):
        from jaxotron.data.synthetic import write_synthetic_dataset
        print("Generating synthetic dataset in data/synthetic ...")
        write_synthetic_dataset("data/synthetic")

    model = MODEL_PRESETS[args.preset]
    meta_vocab = read_meta(args.data).get("vocab_size")
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
        run_name=args.run_name, wandb_project=args.wandb_project, distributed=args.distributed,
        profile_dir=args.profile_dir, dry_run=args.dry_run, show_sharding=args.show_sharding,
    )
    for item in args.set:
        key, _, value = item.partition("=")
        apply_override(cfg, key.strip(), value.strip())

    from jaxotron.trainer import train
    train(cfg)


if __name__ == "__main__":
    main()
