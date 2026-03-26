import numpy as np
import torch


class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError(f"TwoMoons inputs must have shape (N, 2), got {tuple(inputs.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"TwoMoons targets must have shape (N,), got {tuple(targets.shape)}")
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Inputs and targets must have same length")

        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return {
            'input': self.inputs[index],
            'target': self.targets[index],
            'index': index,
        }


def TwoMoons(config, logger):
    """Synthetic two-moons dataset.

    Returns the same dict structure as MNIST/CIFAR loaders:
    - train_dset: torch Dataset yielding {'input','target','index'}
    - test_loader: DataLoader over test set

    Config options (dataset.*):
    - n_samples (int): total samples (default 10000)
    - noise (float): make_moons noise (default 0.15)
    - test_size (float): fraction for test split (default 0.2)
    - standardize (bool): z-score features using train stats (default True)
    - random_state (int|None): overrides config['seed'] if set
    """

    try:
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
    except Exception as e:
        raise ImportError(
            "TwoMoons requires scikit-learn. Install it (e.g., `pip install scikit-learn`)."
        ) from e

    dcfg = config.get('dataset', {})
    seed = dcfg.get('random_state', config.get('seed', 42))

    n_samples = int(dcfg.get('n_samples', 10_000))
    noise = float(dcfg.get('noise', 0.15))
    test_size = float(dcfg.get('test_size', 0.2))
    standardize = bool(dcfg.get('standardize', True))

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    if standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_dset = wrapped_dataset(X_train_t, y_train_t)
    test_dset = wrapped_dataset(X_test_t, y_test_t)

    config['training_opt']['test_batch_size'] = (
        config['training_opt']['batch_size']
        if 'test_batch_size' not in config['training_opt']
        else config['training_opt']['test_batch_size']
    )

    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=config['training_opt']['test_batch_size'],
        shuffle=False,
        num_workers=config['training_opt']['num_data_workers'],
        pin_memory=True,
        drop_last=False,
    )

    return {
        'num_classes': 2,
        'train_dset': train_dset,
        'train_dset_unaugmented': train_dset,
        'test_loader': test_loader,
        'num_train_samples': len(train_dset),
    }
