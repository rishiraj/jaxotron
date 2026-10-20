"""Thin wrapper so `python train.py ...` works from a git checkout. See jaxotron/cli/train.py."""
from jaxotron.cli.train import main

if __name__ == "__main__":
    main()