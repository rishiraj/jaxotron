# Contributing to Jaxotron

Thanks for helping! Jaxotron's goal is to be the most *readable* 4D-parallel training codebase, so clarity beats cleverness.

## Development setup

```bash
git clone https://github.com/rishiraj/jaxotron && cd jaxotron
pip install -e ".[dev,hf]"
pytest -q -m "not slow"   # fast checks (~2-3 min on a laptop)
pytest -q                 # full suite, including end-to-end 4D training
ruff check .
```

The tests run on 16 simulated CPU devices (`tests/conftest.py`), so no accelerator is needed.
The Hugging Face parity test also needs PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Ground rules

- **Every parallel layout must match a single device.** If you touch the model, sharding rules or pipeline, add or keep a case in `tests/test_parallelism.py`.
- **Keep the core small.** New dependencies go into an optional extra. The five core dependencies are a feature.
- **No experimental APIs.** Prefer stable JAX APIs that work from the minimum supported version (see the CI matrix).
- **Comments explain *why*.** The code is read on slides; keep functions short and names obvious.

## Good first contributions

- A model preset or a hardware entry in the MFU table (`jaxotron/utils.py`).
- A KV-cached sampler for `generate`.
- A notebook or slide deck from your own workshop (link it in the README).
- Benchmarks on real hardware (tokens/s and MFU per layout).

## Releasing (maintainers)

1. Update the version in `pyproject.toml`, `jaxotron/__init__.py`, `CITATION.cff` and `CHANGELOG.md`.
2. Tag `vX.Y.Z` and push; `.github/workflows/release.yml` builds and publishes to PyPI with trusted publishing.
