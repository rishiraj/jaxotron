"""Sample text from a Jaxotron checkpoint or a Hugging Face Llama / SmolLM2 / Qwen3 / Qwen2.5 model.

The checkpoint can come from ANY parallel layout: Orbax reshards it onto the
inference mesh here (one device by default, or --tp / --pp / --cp devices).
Uses a fast KV-cached sampler by default (for single-device and tensor-parallel
inference), and automatically falls back to full-sequence forward when --pp > 1,
--cp > 1, or --no_kv_cache is passed.

jaxotron-generate --checkpoint_dir checkpoints/run --prompt "Once upon a time"
jaxotron-generate --checkpoint_dir checkpoints/run --prompt "5 8 11"            # synthetic data: token ids
jaxotron-generate --hf HuggingFaceTB/SmolLM2-135M --prompt "Gravity is"        # pretrained weights
jaxotron-generate --hf Qwen/Qwen3-0.6B --cpu_devices 4 --tp 4 --prompt "JAX is" # Qwen3 + TP inference
"""
from __future__ import annotations

import argparse
import math
import sys
import time

from jaxotron.cli.train import setup_cpu_devices


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Sample from a Jaxotron checkpoint or a Hugging Face Llama/Qwen model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--checkpoint_dir", default="checkpoints/run")
    src.add_argument("--hf", default=None, metavar="REPO_OR_DIR",
                     help="Hugging Face Llama / SmolLM2 / Qwen3 / Qwen2.5 model")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top_k", type=int, default=50, help="0 disables")
    ap.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling; 1.0 disables")
    ap.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer (default: from the model / dataset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"])
    ap.add_argument("--no_kv_cache", action="store_true", help="disable KV cache (re-run full forward per token)")
    ap.add_argument("--cpu", action="store_true", help="run on CPU even if an accelerator is present")
    ap.add_argument("--cpu_devices", type=int, default=0, help="simulate N devices on CPU")
    for axis, desc in (("pp", "pipeline"), ("cp", "context"), ("tp", "tensor")):
        ap.add_argument(f"--{axis}", type=int, default=1, help=f"{desc} parallel size for inference")
    ap.add_argument("--compile_cache", default="~/.cache/jaxotron/xla")
    args = ap.parse_args(argv)
    if args.max_new_tokens < 1:
        ap.error("--max_new_tokens must be >= 1")
    return args


def _sample_from_logits(logits, valid_vocab: int, temperature: float, top_k: int, top_p: float, key):
    import jax
    import jax.numpy as jnp

    logits = jnp.where(jnp.arange(logits.shape[-1]) < valid_vocab, logits, -jnp.inf)  # drop padded vocab
    if temperature <= 0:
        return jnp.argmax(logits)
    logits = logits / temperature
    if top_k:
        kth = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))[0][-1]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    if top_p < 1.0:
        sorted_logits = jnp.sort(logits)[::-1]
        cumulative = jnp.cumsum(jax.nn.softmax(sorted_logits))
        cutoff = sorted_logits[jnp.argmax(cumulative >= top_p)]
        logits = jnp.where(logits < cutoff, -jnp.inf, logits)
    return jax.random.categorical(key, logits)


def main(argv=None):
    args = parse_args(argv)
    if args.cpu_devices:
        setup_cpu_devices(args.cpu_devices)
    elif args.cpu:
        import os
        os.environ["JAX_PLATFORMS"] = "cpu"

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaxotron.checkpoint import Checkpointer
    from jaxotron.config import Config, ParallelConfig, TrainConfig, model_errors
    from jaxotron.data.loader import read_meta
    from jaxotron.models.llama import forward, forward_cached, init_kv_cache
    from jaxotron.parallel.mesh import create_mesh
    from jaxotron.trainer import abstract_model, make_context
    from jaxotron.utils import enable_compilation_cache

    enable_compilation_cache(args.compile_cache)
    pcfg = ParallelConfig(pp=args.pp, cp=args.cp, tp=args.tp, pp_microbatches=1)
    if pcfg.world_size > jax.device_count():
        raise SystemExit(f"--pp*--cp*--tp = {pcfg.world_size} but only {jax.device_count()} devices "
                         f"(use --cpu_devices {pcfg.world_size} on a laptop).")
    mesh = create_mesh(pcfg, jax.devices()[:pcfg.world_size])

    tok_name = args.tokenizer
    if args.hf:
        from jaxotron.hf import hf_model_config, load_hf_llama, resolve_hf_path
        path = resolve_hf_path(args.hf)
        model_cfg = hf_model_config(path)
        cfg = Config(parallel=pcfg, model=model_cfg, train=TrainConfig(dtype=args.dtype, remat="none"))
        errors = model_errors(model_cfg, pcfg)
        if errors:
            raise SystemExit("Invalid layout for this model:\n  - " + "\n  - ".join(errors))
        params, _ = load_hf_llama(path, mesh, model_cfg)
        _, static, _ = abstract_model(model_cfg, mesh)
        tok_name = tok_name or path
        print(f"Loaded {args.hf} ({pcfg.describe()})", file=sys.stderr)
    else:
        cfg = Config.load_json(f"{args.checkpoint_dir.rstrip('/')}/config.json")
        cfg.parallel, cfg.train.dtype, cfg.train.remat = pcfg, args.dtype, "none"
        errors = model_errors(cfg.model, pcfg)
        if errors:
            raise SystemExit("Invalid layout for this model:\n  - " + "\n  - ".join(errors))
        params_abs, static, _ = abstract_model(cfg.model, mesh)
        ckpt = Checkpointer(args.checkpoint_dir)
        try:
            step, params = ckpt.restore_params(params_abs)
        finally:
            ckpt.close()
        tok_name = tok_name or read_meta(cfg.data_path).get("tokenizer") or cfg.init_from_hf
        print(f"Loaded step {step} from {args.checkpoint_dir} ({pcfg.describe()})", file=sys.stderr)

    ctx = make_context(cfg, mesh)
    model = eqx.combine(params, static)

    tokenizer, eos, valid_vocab = None, None, cfg.model.vocab_size
    if tok_name:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise SystemExit('Text prompts need a tokenizer: pip install "jaxotron[hf]"') from None
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        prompt = tokenizer.encode(args.prompt)
        eos, valid_vocab = tokenizer.eos_token_id, min(len(tokenizer), cfg.model.vocab_size)
    else:
        prompt = [int(t) for t in args.prompt.split()] if args.prompt.replace(" ", "").isdigit() else [1, 2]
    if not prompt:
        raise SystemExit("The prompt is empty after tokenisation.")

    first, total = len(prompt), len(prompt) + args.max_new_tokens
    multiple = math.lcm(64, args.cp)
    buf_len = -(-total // multiple) * multiple  # fixed buffer -> compile once
    tokens = np.zeros((1, buf_len), np.int32)
    tokens[0, :first] = prompt

    use_kv_cache = not args.no_kv_cache and args.pp == 1 and args.cp == 1
    sample = lambda logits, key: _sample_from_logits(logits, valid_vocab, args.temperature, args.top_k,
                                                     args.top_p, key)

    def render(ids):
        return tokenizer.decode(ids, skip_special_tokens=True) if tokenizer else " ".join(map(str, ids))

    printed = render(prompt)
    sys.stdout.write(printed)
    sys.stdout.flush()

    def emit(pos: int) -> bool:
        """Stream the newly generated token at `pos`. Returns True if it is EOS."""
        nonlocal printed
        text = render(tokens[0, :pos + 1].tolist())
        if text.startswith(printed):  # stream only the new characters
            sys.stdout.write(text[len(printed):] if tokenizer else " " + str(tokens[0, pos]))
            sys.stdout.flush()
            printed = text
        return eos is not None and tokens[0, pos] == eos

    key = jax.random.PRNGKey(args.seed)
    pos = first
    if use_kv_cache:
        cache = init_kv_cache(cfg.model, batch_size=1, max_seq=buf_len, ctx=ctx)

        @jax.jit
        def prefill(model, prompt_arr, cache, key):
            logits, cache = forward_cached(model, prompt_arr, jnp.int32(0), cache, ctx)
            return sample(logits[0, -1], key), cache

        @jax.jit
        def decode(model, tok_arr, pos, cache, key):
            logits, cache = forward_cached(model, tok_arr, pos, cache, ctx)
            return sample(logits[0, 0], key), cache

        key, sub = jax.random.split(key)
        next_id, cache = prefill(model, jnp.asarray(tokens[:, :first]), cache, sub)
        # Compile the decode step before starting the clock, so tok/s measures decoding rather than XLA.
        jax.block_until_ready(decode(model, jnp.zeros((1, 1), jnp.int32), jnp.int32(first), cache, sub))
        tokens[0, pos] = int(next_id)
        tic = time.perf_counter()
        while not emit(pos) and pos + 1 < total:
            key, sub = jax.random.split(key)
            next_id, cache = decode(model, jnp.asarray(tokens[:, pos:pos + 1]), jnp.int32(pos), cache, sub)
            pos += 1
            tokens[0, pos] = int(next_id)
    else:
        @jax.jit
        def next_token(model, tokens, pos, key):
            return sample(forward(model, tokens, ctx)[0, pos - 1], key)

        tic = None
        while True:
            key, sub = jax.random.split(key)
            tokens[0, pos] = int(next_token(model, jnp.asarray(tokens), pos, sub))
            if tic is None:
                tic = time.perf_counter()  # the first call includes compilation
            if emit(pos) or pos + 1 >= total:
                break
            pos += 1
    end = pos + 1

    final = render(tokens[0, :end].tolist())
    if final != printed:
        sys.stdout.write("\n" + final)
    sys.stdout.write("\n")
    decoded = max(end - first - 1, 1)  # tokens produced after the clock started
    elapsed = max(time.perf_counter() - tic, 1e-6)
    mode = "KV-cache" if use_kv_cache else "full-sequence"
    print(f"[{end - first} tokens | {decoded / elapsed:.1f} tok/s decode ({mode})]", file=sys.stderr)


if __name__ == "__main__":
    main()
