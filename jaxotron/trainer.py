"""The Jaxotron training loop.

The whole training step (gradient accumulation, pipeline schedule, optimizer
update) is a single jax.jit-compiled function. Parallelism is entirely
described by shardings; XLA inserts all communication.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from jaxotron.checkpoint import Checkpointer
from jaxotron.config import Config, ModelConfig
from jaxotron.data.loader import DataLoader, TokenDataset, read_meta, to_global_array
from jaxotron.models.llama import Llama, count_params, flops_per_token, forward
from jaxotron.optim import build_optimizer, init_optimizer_state
from jaxotron.parallel.collectives import collectives_table, count_collectives
from jaxotron.parallel.context import ParallelContext
from jaxotron.parallel.mesh import create_mesh, mesh_table
from jaxotron.parallel.pipeline import bubble_fraction, split_microbatches
from jaxotron.parallel.rules import batch_sharding, param_shardings
from jaxotron.utils import (
    MetricLogger,
    device_memory_gib,
    human,
    peak_flops_per_device,
    print0,
    sharding_table,
)


def _is_array_like(x) -> bool:
    return isinstance(x, (jax.Array, jax.ShapeDtypeStruct))


def _global_norm(tree):
    tree_norm = getattr(getattr(optax, "tree", None), "norm", None)  # optax >= 0.2.5
    return tree_norm(tree) if tree_norm is not None else optax.global_norm(tree)


def resolve_dtype(name: str):
    if name == "auto":
        return jnp.float32 if jax.default_backend() == "cpu" else jnp.bfloat16
    return {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[name]


def make_context(cfg: Config, mesh) -> ParallelContext:
    return ParallelContext(
        mesh=mesh,
        pp=cfg.parallel.pp,
        pp_microbatches=cfg.pp_microbatches,
        sp=cfg.parallel.sp and cfg.parallel.tp > 1,
        dtype=resolve_dtype(cfg.train.dtype),
        remat=cfg.train.remat,
        attention_impl=cfg.train.attention_impl,
    )


def abstract_model(mcfg: ModelConfig, mesh):
    """(abstract sharded params, static part of the model) without allocating memory."""
    abstract = eqx.filter_eval_shape(Llama, mcfg, jax.random.PRNGKey(0))
    params_abs, static = eqx.partition(abstract, _is_array_like)
    shardings = param_shardings(params_abs, mesh)
    params_abs = jax.tree.map(lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), params_abs, shardings)
    return params_abs, static, shardings


def init_model(mcfg: ModelConfig, mesh, key):
    """Initialise parameters directly in their sharded layout: no device ever holds the full model."""
    _, static, shardings = abstract_model(mcfg, mesh)

    def _init(k):
        return eqx.filter(Llama(mcfg, k), eqx.is_array)

    params = jax.jit(_init, out_shardings=shardings)(key)
    return params, static, shardings


def make_loss_fn(static, ctx: ParallelContext) -> Callable:
    def loss_fn(params, inputs, labels):
        logits = forward(eqx.combine(params, static), inputs, ctx)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss_fn


def _split_batch(tokens, accum_steps: int, ctx: ParallelContext):
    inputs, labels = tokens[:, :-1], tokens[:, 1:]
    inputs = ctx.shard(split_microbatches(inputs, accum_steps), None, "batch", "seq")
    labels = ctx.shard(split_microbatches(labels, accum_steps), None, "batch", "seq")
    return inputs, labels


def make_grad_fn(static, ctx: ParallelContext, accum_steps: int) -> Callable:
    """(params, tokens [GBS, T+1]) -> (mean loss, mean grads), with gradient accumulation via lax.scan."""
    value_and_grad = jax.value_and_grad(make_loss_fn(static, ctx))

    def grad_fn(params, tokens):
        inputs, labels = _split_batch(tokens, accum_steps, ctx)
        if accum_steps == 1:
            return value_and_grad(params, inputs[0], labels[0])

        def accumulate(carry, micro):
            loss_sum, grad_sum = carry
            loss, grads = value_and_grad(params, *micro)
            return (loss_sum + loss, jax.tree.map(jnp.add, grad_sum, grads)), None

        zeros = (jnp.zeros((), jnp.float32), jax.tree.map(jnp.zeros_like, params))
        (loss_sum, grad_sum), _ = jax.lax.scan(accumulate, zeros, (inputs, labels))
        return loss_sum / accum_steps, jax.tree.map(lambda g: g / accum_steps, grad_sum)

    return grad_fn


def make_train_step(static, optimizer, ctx: ParallelContext, accum_steps: int) -> Callable:
    grad_fn = make_grad_fn(static, ctx, accum_steps)

    def train_step(params, opt_state, tokens):
        loss, grads = grad_fn(params, tokens)
        grad_norm = _global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"loss": loss, "grad_norm": grad_norm}

    return train_step


def make_eval_step(static, ctx: ParallelContext, accum_steps: int) -> Callable:
    loss_fn = make_loss_fn(static, ctx)

    def eval_step(params, tokens):
        inputs, labels = _split_batch(tokens, accum_steps, ctx)
        losses = jax.lax.map(lambda m: loss_fn(params, *m), (inputs, labels))
        return losses.mean()

    return eval_step


def _check_vocab(cfg: Config) -> None:
    meta_vocab = read_meta(cfg.data_path).get("vocab_size")
    if meta_vocab and meta_vocab > cfg.model.vocab_size:
        raise ValueError(f"Dataset vocab ({meta_vocab}) exceeds model vocab_size ({cfg.model.vocab_size}). "
                         "Tokenise the data with the model's tokenizer (prepare_data.py --tokenizer ...).")


def _print_header(cfg: Config, mesh, ctx: ParallelContext, accum: int) -> None:
    t, p = cfg.train, cfg.parallel
    print0(f"Jaxotron | jax {jax.__version__} | backend={jax.default_backend()} | devices={jax.device_count()} "
           f"({jax.devices()[0].device_kind}) | processes={jax.process_count()}")
    print0(f"Mesh: {dict(mesh.shape)}  ->  {p.describe()}")
    source = f"from {cfg.init_from_hf}" if cfg.init_from_hf else "from scratch"
    print0(f"Model: {human(count_params(cfg.model))} params ({source}) | compute dtype={jnp.dtype(ctx.dtype).name} "
           f"| remat={t.remat}")
    print0(f"Batch: global={t.global_batch_size} x {t.sequence_length} tokens "
           f"({human(t.global_batch_size * t.sequence_length)} tok/step) | micro/replica={t.micro_batch_size} "
           f"| grad accumulation={accum} | pipeline microbatches={cfg.pp_microbatches}")
    if p.pp > 1:
        print0(f"Pipeline bubble: {bubble_fraction(p.pp, cfg.pp_microbatches):.0%} of pipeline time is idle "
               f"(raise micro_batch_size / --pp_microbatches to shrink it)")
        if p.fsdp > 1:
            print0("Note: with pp > 1, prefer dp over fsdp. FSDP re-gathers weights for every pipeline "
                   "microbatch, and XLA may insert extra resharding (results stay correct, just slower).")


def train(cfg: Config) -> dict:
    if cfg.distributed:
        jax.distributed.initialize()

    cfg.validate(jax.device_count())
    _check_vocab(cfg)
    t, p = cfg.train, cfg.parallel
    total_steps = cfg.total_train_steps
    mesh = create_mesh(p)
    ctx = make_context(cfg, mesh)
    accum = cfg.grad_accum_steps
    _print_header(cfg, mesh, ctx, accum)
    if cfg.show_mesh:
        print0(mesh_table(mesh))

    # ---- model & optimizer (created directly in sharded form) ---------------
    ckpt = Checkpointer(cfg.checkpoint_dir, t.keep_checkpoints)
    if cfg.init_from_hf and ckpt.latest_step() is None:
        from jaxotron.hf import load_hf_llama
        _, static, p_shardings = abstract_model(cfg.model, mesh)
        params, _ = load_hf_llama(cfg.init_from_hf, mesh, cfg.model)
        print0(f"Loaded pretrained weights from {cfg.init_from_hf}")
    else:
        params, static, p_shardings = init_model(cfg.model, mesh, jax.random.PRNGKey(t.seed))
    optimizer, schedule = build_optimizer(t, params, total_steps=total_steps)
    opt_state, o_shardings = init_optimizer_state(optimizer, params, p_shardings, mesh)
    if cfg.show_sharding:
        print0(sharding_table(params, opt_state))

    # ---- checkpoint resume --------------------------------------------------
    start_step, params, opt_state = ckpt.restore(params, opt_state)
    if start_step:
        print0(f"Resumed from step {start_step} ({cfg.checkpoint_dir})")
    if jax.process_index() == 0:
        cfg.save_json(f"{cfg.checkpoint_dir.rstrip('/')}/config.json")

    # ---- data ---------------------------------------------------------------
    data_sharding = batch_sharding(mesh)
    train_ds = TokenDataset(cfg.data_path, t.sequence_length)
    loader = DataLoader(train_ds, t.global_batch_size, data_sharding, seed=t.seed, start_step=start_step)

    # ---- compile ahead of time ----------------------------------------------
    first_batch = next(loader)
    tic = time.perf_counter()
    train_step = jax.jit(make_train_step(static, optimizer, ctx, accum),
                         donate_argnums=(0, 1), out_shardings=(p_shardings, o_shardings, None))
    train_step = train_step.lower(params, opt_state, first_batch).compile()
    print0(f"Compiled the train step in {time.perf_counter() - tic:.1f}s")
    if cfg.show_collectives:
        print0("Communication inserted by XLA:\n" + collectives_table(count_collectives(train_step.as_text(), mesh)))

    if cfg.dry_run:
        mem = train_step.memory_analysis()
        if mem is not None:
            for field in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes"):
                if hasattr(mem, field):
                    print0(f"  {field:<24} {getattr(mem, field) / 2**30:8.3f} GiB per device")
        print0("Dry run complete: the step compiles for this mesh.")
        loader.close()
        ckpt.close()
        return {}

    eval_step, val_batches = None, []
    if t.eval_every and cfg.val_path:
        eval_step = jax.jit(make_eval_step(static, ctx, accum))
        val_loader = DataLoader(TokenDataset(cfg.val_path, t.sequence_length), t.global_batch_size,
                                data_sharding, shuffle=False, prefetch=1)
        val_batches = [val_loader.host_batch(i) for i in range(t.eval_batches)]  # fixed held-out set
        val_loader.close()

    logger = MetricLogger(cfg.wandb_project, cfg.run_name, cfg.to_dict())
    tokens_per_step = t.global_batch_size * t.sequence_length
    flops_per_step = flops_per_token(cfg.model, t.sequence_length) * tokens_per_step
    peak = cfg.peak_tflops * 1e12 if cfg.peak_tflops else (peak_flops_per_device() if ctx.dtype == jnp.bfloat16 else None)

    print0("Starting training...")
    last, done, last_saved, profiling = {}, start_step, start_step, False
    timer, last_logged = time.perf_counter(), start_step
    try:
        for step in range(start_step, total_steps):
            if cfg.profile_dir and step == start_step + 3:
                jax.profiler.start_trace(cfg.profile_dir)
                profiling = True

            batch = first_batch if step == start_step else next(loader)
            params, opt_state, metrics = train_step(params, opt_state, batch)
            done = step + 1

            if profiling and (step == start_step + 6 or done == total_steps):
                jax.block_until_ready(params)
                jax.profiler.stop_trace()
                profiling = False
                print0(f"Profile written to {cfg.profile_dir} (open with TensorBoard or https://ui.perfetto.dev)")

            if done % t.log_every == 0 or step == start_step or done == total_steps:
                metrics = jax.device_get(metrics)
                loss = float(metrics["loss"])
                if not math.isfinite(loss):
                    raise FloatingPointError(f"Loss is {loss} at step {done}. Try a lower --lr, more --warmup, "
                                             "or --dtype float32. The last good checkpoint is kept.")
                elapsed = max(time.perf_counter() - timer, 1e-9)
                n = max(done - last_logged, 1)
                tok_s = n * tokens_per_step / elapsed
                trained_tokens = done * tokens_per_step
                line = (f"step {done:>6} | loss {loss:.4f} | grad_norm {float(metrics['grad_norm']):.3f} "
                        f"| lr {float(schedule(step)):.2e} | {human(tok_s)} tok/s "
                        f"| {human(tok_s / jax.device_count())} tok/s/device "
                        f"| tokens {human(trained_tokens)}")
                log = {"train/loss": loss, "train/grad_norm": float(metrics["grad_norm"]),
                       "train/lr": float(schedule(step)), "perf/tokens_per_sec": tok_s,
                       "train/tokens": trained_tokens}
                if peak and step != start_step:
                    mfu = n * flops_per_step / elapsed / (peak * jax.device_count())
                    line += f" | MFU {mfu:.1%}"
                    log["perf/mfu"] = mfu
                mem_gib = device_memory_gib()
                if mem_gib is not None:
                    line += f" | mem {mem_gib:.2f} GiB"
                    log["perf/memory_gib"] = mem_gib
                print0(line)
                logger.log(log, step=done)
                last = log
                timer, last_logged = time.perf_counter(), done

            if eval_step is not None and done % t.eval_every == 0:
                eval_tic = time.perf_counter()
                losses = [eval_step(params, to_global_array(b, data_sharding)) for b in val_batches]
                val_loss = float(jnp.mean(jnp.stack(losses)))
                print0(f"step {done:>6} | val_loss {val_loss:.4f}")
                logger.log({"val/loss": val_loss}, step=done)
                last["val/loss"] = val_loss
                timer += time.perf_counter() - eval_tic  # exclude eval time without losing step timing

            if done % t.save_every == 0 or done == total_steps:
                ckpt.save(done, params, opt_state)
                last_saved = done
    except KeyboardInterrupt:
        if jax.process_count() > 1:
            raise
        print0(f"\nInterrupted. Saving step {done} so the same command resumes from here...")
        if done > last_saved:
            try:
                ckpt.save(done, params, opt_state)
            except Exception as e:  # e.g. interrupted mid-step, buffers already donated
                print0(f"Could not save step {done} ({e}); resuming from step {last_saved} instead.")
    finally:
        if profiling:
            jax.profiler.stop_trace()
        ckpt.close()
        loader.close()
        logger.finish()

    print0(f"Done. Checkpoints in {ckpt.directory}")
    return last