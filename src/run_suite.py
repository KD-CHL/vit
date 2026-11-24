import argparse
import subprocess
import sys

from .common import ROOT, SEEDS
from .train import CONFIG_FILES


def run(module, *arguments):
    subprocess.run(
        [sys.executable, "-u", "-m", module, *map(str, arguments)],
        cwd=ROOT, check=True,
    )


def train_run(base, seed, workers, name=None, **overrides):
    if overrides:
        from .train import default_config, train

        config = default_config(base)
        config.update(overrides)
        train(config, seed, name, workers)
    else:
        path = ROOT / "configs" / f"{CONFIG_FILES[base]}.yaml"
        run("src.train", "--config", path, "--seed", seed, "--workers", workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--ablation", choices=["v2_mean", "v3_none", "v3_basic", "v4_r4", "v4_r16"])
    args = parser.parse_args()
    ablations = {
        "v2_mean": ("v2", {"pool": "mean"}),
        "v3_none": ("v3", {"augmentation": "none"}),
        "v3_basic": ("v3", {"augmentation": "basic"}),
        "v4_r4": ("v4", {"rank": 4}),
        "v4_r16": ("v4", {"rank": 16}),
    }
    if args.ablation:
        experiment, overrides = ablations[args.ablation]
        train_run(experiment, SEEDS[0], args.workers, f"{args.ablation}_s{SEEDS[0]}",
                  **overrides)
        return
    for seed in SEEDS:
        for experiment in ("v3", "v4", "v2", "v1"):
            train_run(experiment, seed, args.workers)
    for ablation in ablations:
        run("src.run_suite", "--workers", args.workers, "--ablation", ablation)
    if not args.train_only:
        run("src.evaluate", "--workers", args.workers)
        run("src.visualize")
        run("src.summarize")


if __name__ == "__main__":
    main()
