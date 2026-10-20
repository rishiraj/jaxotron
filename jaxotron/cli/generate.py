"""Sample text from a Jaxotron checkpoint or a Hugging Face Llama model.

The checkpoint can come from ANY parallel layout: Orbax reshards it onto the
inference mesh here (one device by default, or --tp / --pp / --cp devices).
This is a simple educational sampler: no KV cache, it re-runs the forward pass
over a fixed-size buffer for every new token, which is fine for small models.

jaxotron-generate --checkpoint_dir checkpoints/run --prompt "Once upon a time"
jaxotron-generate --checkpoint_dir checkpoints/run --prompt "5 8 11"            # synthetic data: token ids
jaxotron-generate --hf HuggingFaceTB/SmolLM2-135M --prompt "Gravity is"        # pretrained weights
jaxotron-generate --hf HuggingFaceTB/SmolLM2-135M --cpu_devices 3 --tp 3       # tensor-parallel inference
"""
from __future__ import annotations

import argparse
import math
import sys

from jaxotron.cli.train import setup_cpu_devices


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Sample from a Jaxotron checkpoint or a Hugging Face Llama model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--checkpoint_dir", default="checkpoints/run")
    src.add_argument("--hf", default=None, metavar="REPO_OR_DIR", help="Hugging Face Llama-family model")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top_k", type=int, default=50, help="0 disables")
    ap.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling; 1.0 disables")
    ap.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer (default: from the model / dataset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"])
    ap.add_argument("--cpu", action="store_true", help="run on CPU even if an accelerator is present")
    ap.add_argument("--cpu_devices", type=int, default=0, help="simulate N devices on CPU")
    for axis, desc in (("pp", "pipeline"), ("cp", "context"), ("tp", "tensor")):
        ap.add_argument(f"--{axis}", type=int, default=1, help=f"{desc} parallel size for inference")
    ap.add_argument("--compile_cache", default="~/.cache/jaxotron/xla")
    return ap.parse_args(argv)


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
    from jaxotron.models.llama import forward
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
        print(f"Loaded {args.hf}", file=sys.stderr)
    else:
        cfg = Config.load_json(f"{args.checkpoint_dir.rstrip('/')}/config.json")
        cfg.parallel, cfg.train.dtype, cfg.train.remat = pcfg, args.dtype, "none"
        errors = model_errors(cfg.model, pcfg)
        if errors:
            raise SystemExit("Invalid layout for this model:\n  - " + "\n  - ".join(errors))
        params_abs, static, _ = abstract_model(cfg.model, mesh)
        step, params = Checkpointer(args.checkpoint_dir).restore_params(params_abs)
        tok_name = tok_name or read_meta(cfg.data_path).get("tokenizer") or cfg.init_from_hf
        print(f"Loaded step {step} from {args.checkpoint_dir}", file=sys.stderr)

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

    total = len(prompt) + args.max_new_tokens
    multiple = math.lcm(64, args.cp)
    buf_len = -(-total // multiple) * multiple  # fixed buffer -> compile once
    tokens = np.zeros((1, buf_len), np.int32)
    tokens[0, :len(prompt)] = prompt

    @jax.jit
    def next_token(model, tokens, pos, key):
        logits = forward(model, tokens, ctx)[0, pos - 1]
        logits = jnp.where(jnp.arange(logits.shape[-1]) < valid_vocab, logits, -jnp.inf)  # drop padded vocab
        if args.temperature <= 0:
            return jnp.argmax(logits)
        logits = logits / args.temperature
        if args.top_k:
            kth = jax.lax.top_k(logits, min(args.top_k, logits.shape[-1]))[0][-1]
            logits = jnp.where(logits < kth, -jnp.inf, logits)
        if args.top_p < 1.0:
            sorted_logits = jnp.sort(logits)[::-1]
            cumulative = jnp.cumsum(jax.nn.softmax(sorted_logits))
            cutoff = sorted_logits[jnp.argmax(cumulative >= args.top_p)]
            logits = jnp.where(logits < cutoff, -jnp.inf, logits)
        return jax.random.categorical(key, logits)

    def render(ids):
        return tokenizer.decode(ids, skip_special_tokens=True) if tokenizer else " ".join(map(str, ids))

    key = jax.random.PRNGKey(args.seed)
    printed = render(prompt)
    sys.stdout.write(printed)
    sys.stdout.flush()
    end = total
    for pos in range(len(prompt), total):
        key, sub = jax.random.split(key)
        tokens[0, pos] = int(next_token(model, jnp.asarray(tokens), pos, sub))
        text = render(tokens[0, :pos + 1].tolist())
        if text.startswith(printed):  # stream only the new characters
            sys.stdout.write(text[len(printed):] if tokenizer else " " + str(tokens[0, pos]))
            sys.stdout.flush()
            printed = text
        if eos is not None and tokens[0, pos] == eos:
            end = pos + 1
            break
    final = render(tokens[0, :end].tolist())
    if final != printed:
        sys.stdout.write("\n" + final)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
