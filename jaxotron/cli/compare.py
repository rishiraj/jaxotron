"""Compare all parallelism strategies side-by-side in one table (`jaxotron-compare`).

Runs one forward + backward step across 8 layouts (Single device, DP, FSDP,
TP, TP+SP, CP, PP, and Full 4D) on the exact same batch and random seed, and
prints per-device parameter memory, XLA collectives, and max gradient difference
against the single-device baseline. Ideal for conference talks and sanity checks.

On a machine with 8 accelerators (e.g. a Kaggle TPU v5e-8) the 16-device 4D row
is replaced by a 3D (PP x CP x TP) row.
"""
from __future__ import annotations

import argparse
import math

from jaxotron.cli.train import setup_cpu_devices


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu_devices", type=int, default=16, help="simulated CPU devices (0 uses real devices)")
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--global_batch_size", type=int, default=16)
    ap.add_argument("--compile_cache", default="~/.cache/jaxotron/xla")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cpu_devices:
        setup_cpu_devices(args.cpu_devices)

    import jax
    import numpy as np

    from jaxotron.config import Config, ModelConfig, ParallelConfig, TrainConfig
    from jaxotron.data.loader import to_global_array
    from jaxotron.parallel.collectives import COLLECTIVES, count_collectives
    from jaxotron.parallel.mesh import create_mesh
    from jaxotron.parallel.rules import batch_sharding
    from jaxotron.trainer import init_model, make_context, make_grad_fn
    from jaxotron.utils import enable_compilation_cache

    # TPUs (and TF32 GPUs) run float32 matmuls in reduced precision by default. Full precision makes
    # the parity column measure the parallel layout, not the matmul unit.
    jax.config.update("jax_default_matmul_precision", "highest")
    enable_compilation_cache(args.compile_cache)
    model = ModelConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4)
    tokens = np.random.default_rng(0).integers(0, model.vocab_size, (args.global_batch_size, args.seq_len + 1), np.int32)

    layouts = [
        ("1-device baseline", dict()),
        ("DP = 4", dict(dp=4)),
        ("FSDP = 4 (ZeRO-3)", dict(fsdp=4)),
        ("TP = 4", dict(tp=4)),
        ("TP = 4 + SP", dict(tp=4, sp=True)),
        ("CP = 4 (seq split)", dict(cp=4)),
        ("PP = 4 (GPipe)", dict(pp=4)),
        ("4D (2x2x2x2)", dict(dp=2, pp=2, cp=2, tp=2)),
    ]
    if 8 <= jax.device_count() < 16:
        layouts[-1] = ("3D PPxCPxTP (2x2x2)", dict(pp=2, cp=2, tp=2))

    rows = [("strategy", "devices", "KB/device", "loss", "max |Δgrad|", "XLA collectives (mesh axis)")]
    ref_grads, skipped = None, []

    for label, kw in layouts:
        pcfg = ParallelConfig(**kw)
        if pcfg.world_size > jax.device_count():
            skipped.append(f"{label} (needs {pcfg.world_size} devices)")
            continue
        cfg = Config(
            parallel=pcfg,
            model=model,
            train=TrainConfig(
                global_batch_size=args.global_batch_size,
                micro_batch_size=args.global_batch_size // pcfg.data_parallel_size,
                sequence_length=args.seq_len,
                dtype="float32",
            ),
        )
        mesh = create_mesh(pcfg, jax.devices()[:pcfg.world_size])
        ctx = make_context(cfg, mesh)
        params, static, _ = init_model(model, mesh, jax.random.PRNGKey(0))
        param_kb = sum(
            math.prod(leaf.sharding.shard_shape(leaf.shape)) * leaf.dtype.itemsize
            for leaf in jax.tree.leaves(params)
        ) / 1024.0
        batch = to_global_array(tokens, batch_sharding(mesh))
        compiled = jax.jit(make_grad_fn(static, ctx, 1)).lower(params, batch).compile()
        loss, grads = compiled(params, batch)
        grads_np = jax.tree.map(np.asarray, grads)
        if ref_grads is None:
            ref_grads = grads_np
            max_diff = 0.0
        else:
            max_diff = max(
                float(np.max(np.abs(g - r)))
                for g, r in zip(jax.tree.leaves(grads_np), jax.tree.leaves(ref_grads))
            )
        stats = count_collectives(compiled.as_text(), mesh)
        summary = ", ".join(
            f"{op}({'+'.join(ax) or '-'})x{s['count']}"
            for (op, ax), s in sorted(stats.items(), key=lambda kv: (COLLECTIVES.index(kv[0][0]), kv[0][1]))
        ) or "none"
        rows.append((
            label,
            str(pcfg.world_size),
            f"{param_kb:7.1f}",
            f"{float(loss):.5f}",
            f"{max_diff:.2e}",
            summary,
        ))

    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    header = "  ".join(r.ljust(w) for r, w in zip(rows[0], widths))
    print(header)
    print("-" * len(header))
    for r in rows[1:]:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    if skipped:
        print(f"Skipped: {', '.join(skipped)}. Use --cpu_devices 16 to simulate them.")
    print("(KB/device counts parameters only; AdamW state follows the same sharding.)")


if __name__ == "__main__":
    main()
