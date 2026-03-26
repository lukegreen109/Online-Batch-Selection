import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import data
from utils import get_configs


class _SimpleLogger:
    def info(self, msg):
        print(msg)


def init_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _batch_to_labels(batch):
    y = batch["target"]
    if isinstance(y, torch.Tensor):
        if y.ndim == 1:
            return y.cpu().numpy().astype(np.int32)
        if y.ndim == 2:
            return y.argmax(-1).cpu().numpy().astype(np.int32)

    y = np.asarray(y)
    if y.ndim == 2:
        y = y.argmax(-1)
    return y.astype(np.int32)


def collect_labels(loader):
    ys = []
    for batch in loader:
        ys.append(_batch_to_labels(batch))

    y = np.concatenate(ys, axis=0)

    try:
        n_dataset = len(loader.dataset)
        if y.shape[0] != n_dataset:
            raise ValueError(f"Collected {y.shape[0]} labels, dataset len {n_dataset}")
    except TypeError:
        # Some iterable datasets may not expose __len__.
        pass

    return y


def save_labels(train_loader, val_loader, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    y_train = collect_labels(train_loader)
    y_val = collect_labels(val_loader)
    payload = {"train": y_train, "val": y_val}
    torch.save(payload, out_path)
    print(f"Saved labels: train={y_train.shape[0]}, val={y_val.shape[0]} -> {out_path}")


def build_config(args):
    data_config = get_configs(args.data)
    optim_config = get_configs(args.optim) if args.optim is not None else {}

    config = {**data_config, **optim_config}
    if "training_opt" not in config:
        config["training_opt"] = {}

    config["seed"] = args.seed
    config["training_opt"]["batch_size"] = args.batch_size
    config["training_opt"]["num_data_workers"] = args.num_workers
    return config


def main():
    parser = argparse.ArgumentParser(description="Save train/val labels once for a dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset config yaml")
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Optional path to optimizer/training config yaml",
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name in output filename")
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite if output file exists")
    args = parser.parse_args()

    config = build_config(args)
    init_seeds(args.seed)

    logger = _SimpleLogger()
    dataset_name = config["dataset"]["name"]
    dataset_fn = getattr(data, dataset_name)
    data_info = dataset_fn(config, logger)

    train_loader = DataLoader(
        data_info["train_dset"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = data_info["test_loader"]

    output_dataset_name = args.dataset if args.dataset is not None else dataset_name
    out_path = args.output if args.output is not None else f"results/data/labels_{output_dataset_name}.p"

    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(
            f"Output already exists at {out_path}. Use --overwrite to replace it."
        )

    save_labels(train_loader, val_loader, out_path)


if __name__ == "__main__":
    main()