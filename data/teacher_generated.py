import os
from typing import Dict, Any

import torch

import models


class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.ndim != 2:
            raise ValueError(
                f"Teacher_Generated inputs must have shape (N, D), got {tuple(inputs.shape)}"
            )
        if targets.ndim != 1:
            raise ValueError(
                f"Teacher_Generated targets must have shape (N,), got {tuple(targets.shape)}"
            )
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


def _strip_module_prefix_if_needed(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith('module.'):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            return checkpoint['state_dict']
        if 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
            return checkpoint['model_state_dict']
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Unsupported teacher checkpoint format. Expected a raw state_dict, or a dict with "
        "'state_dict' / 'model_state_dict'."
    )


def _build_teacher_model(config, logger, device: torch.device):
    dataset_cfg = config.get('dataset', {})
    teacher_model_path = dataset_cfg.get('generating_teacher_model_path', None)
    if teacher_model_path is None or str(teacher_model_path).strip() == '':
        raise ValueError(
            "dataset.generating_teacher_model_path must be provided for Teacher_Generated."
        )
    if not os.path.isfile(teacher_model_path):
        raise FileNotFoundError(
            f"Teacher model file not found at: {teacher_model_path}"
        )

    model_type = dataset_cfg['networks']['type']
    model_args = dict(dataset_cfg['networks'].get('params', {}))
    model_args['in_channels'] = dataset_cfg['in_channels']
    model_args['num_classes'] = dataset_cfg['num_classes']

    try:
        teacher_model = getattr(models, model_type)(**model_args)
    except AttributeError as e:
        raise ValueError(f"Unknown model type for teacher generation: {model_type}") from e

    checkpoint = torch.load(teacher_model_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_module_prefix_if_needed(state_dict)
    teacher_model.load_state_dict(state_dict)
    teacher_model.to(device)
    teacher_model.eval()
    logger.info(f"Loaded teacher model for data generation from {teacher_model_path}")
    return teacher_model


def Teacher_Generated(config, logger):
    """Synthetic dataset generated from a pretrained teacher model.

    Generation rule:
    X = (torch.rand(n_samples, in_channels) - 0.5) * 2.0
    Y = softmax(teacher(X) / tau)
    labels = argmax(Y)
    """

    dcfg = config.get('dataset', {})
    seed = int(config.get('seed', 42))
    n_samples = int(dcfg.get('n_samples', 1000))
    test_size = float(dcfg.get('test_size', 0.2))
    tau = float(dcfg.get('tau', 1.0))
    in_channels = int(dcfg['in_channels'])
    num_classes = int(dcfg['num_classes'])

    if n_samples < 2:
        raise ValueError("Teacher_Generated requires n_samples >= 2")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = _build_teacher_model(config, logger, device)

    X = (torch.rand(n_samples, in_channels, generator=generator) - 0.5) * 2.0

    with torch.no_grad():
        logits = teacher_model(X.to(device))
        Y = torch.softmax(logits / tau, dim=-1)
        predicted_labels = torch.argmax(Y, dim=-1)

    predicted_labels = predicted_labels.to(torch.long).cpu()
    X = X.cpu().to(torch.float32)

    if predicted_labels.max().item() >= num_classes or predicted_labels.min().item() < 0:
        raise ValueError(
            "Generated labels are outside configured [0, num_classes-1] range. "
            "Check dataset.num_classes and the teacher checkpoint."
        )

    num_test = max(1, min(n_samples - 1, int(n_samples * test_size)))
    perm = torch.randperm(n_samples, generator=generator)
    test_indices = perm[:num_test]
    train_indices = perm[num_test:]

    X_train = X[train_indices]
    y_train = predicted_labels[train_indices]
    X_test = X[test_indices]
    y_test = predicted_labels[test_indices]

    train_dset = wrapped_dataset(X_train, y_train)
    test_dset = wrapped_dataset(X_test, y_test)

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

    logger.info(
        "Teacher_Generated dataset created: "
        f"train={len(train_dset)}, test={len(test_dset)}, "
        f"n_samples={n_samples}, in_channels={in_channels}, tau={tau}"
    )

    return {
        'num_classes': num_classes,
        'train_dset': train_dset,
        'train_dset_unaugmented': train_dset,
        'test_loader': test_loader,
        'num_train_samples': len(train_dset),
    }