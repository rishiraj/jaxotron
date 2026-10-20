"""Sample text from a Jaxotron checkpoint.

The checkpoint can come from ANY parallel layout: Orbax reshards it onto a
single device here. (Simple educational sampler: no KV cache, it re-runs the
full forward pass for every new token, which is fine for small models.)

python generate.py --checkpoint_dir checkpoints/run --prompt "Once upon a time"
python generate.py --checkpoint_dir checkpoints/run --prompt "5 8 11"   # synthetic data: token ids
"""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description="Sample from a Jaxotron checkpoint")
    ap.add_argument("--checkpoint_dir", default="checkpoints/run")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer (default: from dataset meta.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx

    from jaxotron.checkpoint import Checkpointer
    from jaxotron.config import Config, ParallelConfig
    from jaxotron.data.loader import read_meta
    from jaxotron.models.llama import forward
    from jaxotron.parallel.mesh import create_mesh
    from jaxotron.trainer import abstract_model, make_context

    cfg = Config.load_json(f"{args.checkpoint_dir.rstrip('/')}/config.json")
    cfg.parallel = ParallelConfig()  # inference on one device
    mesh = create_mesh(cfg.parallel, jax.devices()[:1])
    ctx = make_context(cfg, mesh)
    params_abs, static, _ = abstract_model(cfg.model, mesh)
    step, params = Checkpointer(args.checkpoint_dir).restore_params(params_abs)
    model = eqx.combine(params, static)
    print(f"Loaded step {step} from {args.checkpoint_dir}")

    tok_name = args.tokenizer or read_meta(cfg.data_path).get("tokenizer")
    tokenizer = None
    if tok_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        prompt = tokenizer.encode(args.prompt)
    else:
        prompt = [int(t) for t in args.prompt.split()] if args.prompt.replace(" ", "").isdigit() else [1, 2]

    total = len(prompt) + args.max_new_tokens
    buf_len = -(-total // 64) * 64  # fixed buffer -> compile once
    tokens = np.zeros((1, buf_len), np.int32)
    tokens[0, :len(prompt)] = prompt

    @jax.jit
    def next_token(model, tokens, pos, key):
        logits = forward(model, tokens, ctx)[0, pos - 1]
        if args.temperature <= 0:
            return jnp.argmax(logits)
        logits = logits / args.temperature
        if args.top_k:
            kth = jax.lax.top_k(logits, args.top_k)[0][-1]
            logits = jnp.where(logits < kth, -jnp.inf, logits)
        return jax.random.categorical(key, logits)

    key = jax.random.PRNGKey(args.seed)
    for pos in range(len(prompt), total):
        key, sub = jax.random.split(key)
        tokens[0, pos] = int(next_token(model, jnp.asarray(tokens), pos, sub))

    out = tokens[0, :total].tolist()
    print(tokenizer.decode(out) if tokenizer else " ".join(map(str, out)))


if __name__ == "__main__":
    main()
