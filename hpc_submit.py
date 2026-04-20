"""
hpc_submit.py — submit one SLURM job per config combination.

Usage:
  python hpc_submit.py \\
    --method configs/method/uniform-0.1.yaml \\
    --data   configs/data/mnist.yaml \\
    --model  configs/model/smallcnn.yaml \\
    --optim  configs/optim/sgd-smoke.yaml \\
    --seed   42

  # Multiple values → one job per combination (parallel across nodes)
  python hpc_submit.py \\
    --method configs/method/uniform-0.1.yaml configs/method/divbs-0.1.yaml \\
    --data   configs/data/cifar10.yaml \\
    --model  configs/model/resnet18.yaml \\
    --optim  configs/optim/adam-320-0.001-0.0.yaml \\
    --seed   42 7

  # Dry run — write scripts without submitting
  python hpc_submit.py ... --dry-run
"""

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def make_script(slurm, experiment_cmd):
    lines = [
        "#!/bin/bash",
        f"#SBATCH --account={slurm['account']}",
        f"#SBATCH --gres=gpu:1",
        f"#SBATCH --mem={slurm.get('mem', '32G')}",
        f"#SBATCH --time={slurm.get('time', '08:00:00')}",
        "",
        experiment_cmd,
    ]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hpc",    default="configs/supercomputer/default.yaml")
    p.add_argument("--method", nargs="+", required=True)
    p.add_argument("--data",   nargs="+", required=True)
    p.add_argument("--model",  nargs="+", required=True)
    p.add_argument("--optim",  nargs="+", required=True)
    p.add_argument("--vis",    default=None)
    p.add_argument("--seed",   type=int, nargs="+", default=[42])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    hpc   = load_yaml(args.hpc)
    slurm = hpc["slurm"]

    combos = list(itertools.product(args.method, args.data, args.model, args.optim, args.seed))
    print(f"{len(combos)} job(s) — {'dry run' if args.dry_run else 'submitting'}\n")

    for i, (method, data, model, optim, seed) in enumerate(combos, 1):
        name = f"{Path(method).stem}_{Path(data).stem}_s{seed}"

        cmd = (
            f"CUDA_VISIBLE_DEVICES=0 uv run main.py \\\n"
            f"  --method {method} \\\n"
            f"  --data   {data} \\\n"
            f"  --model  {model} \\\n"
            f"  --optim  {optim} \\\n"
            f"  --seed   {seed} \\\n"
            f"  --wandb_not_upload"
        )
        if args.vis:
            cmd = cmd.replace("--wandb_not_upload", f"--vis    {args.vis} \\\n  --wandb_not_upload")

        script = make_script(slurm, cmd)

        print(f"[{i}/{len(combos)}] {name}")

        if args.dry_run:
            print(script)
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sbatch", delete=False) as f:
                f.write(script)
                tmp = f.name
            r = subprocess.run(["sbatch", tmp], capture_output=True, text=True)
            os.unlink(tmp)
            if r.returncode == 0:
                print(f"  {r.stdout.strip()}")
            else:
                print(f"  ERROR: {r.stderr.strip()}", file=sys.stderr)
        print()

    if args.dry_run:
        print("Dry run done. Remove --dry-run to submit.")


if __name__ == "__main__":
    main()
