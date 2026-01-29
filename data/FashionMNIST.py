from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return {
            'input': self.dataset[index][0],
            'target': self.dataset[index][1],
            'index': index
        }

def FashionMNIST(config, logger):
    logger.info("Loading Fashion-MNIST dataset (CIFAR-style dictionary)...")

    dataset_root = config['dataset']['root']
    batch_size = config['training_opt']['batch_size']
    include_holdout = config['dataset'].get('include_holdout', False)
    num_classes = 10

    if 'FashionMNIST' not in mean_std['mean']:
        mean_std['mean']['FashionMNIST'] = [0.2860]
        mean_std['std']['FashionMNIST'] = [0.3530]

    # === Transforms ===
    transform_aug = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),          # match CIFAR dimensions
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_plain = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # === Load datasets ===
    dst_train = datasets.FashionMNIST(root=dataset_root, train=True, transform=transform_aug, download=True)
    dst_train_unaug = datasets.FashionMNIST(root=dataset_root, train=True, transform=transform_plain, download=True)
    dst_test = datasets.FashionMNIST(root=dataset_root, train=False, transform=transform_plain, download=True)

    # === Split if holdout included ===
    if include_holdout:
        n_total = len(dst_train)
        n_holdout = int(0.1 * n_total)
        n_train = n_total - n_holdout
        train_dset, holdout_dset = random_split(dst_train, [n_train, n_holdout])
        train_dset_unaug, _ = random_split(dst_train_unaug, [n_train, n_holdout])
    else:
        train_dset = dst_train
        train_dset_unaug = dst_train_unaug
        holdout_dset = None

    # === Test loader ===
    test_loader = DataLoader(
        wrapped_dataset(dst_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # === Metadata ===
    fashion_classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    fashion_templates = [f"a picture of a {cls}" for cls in fashion_classes]

    # === Return CIFAR-style dictionary ===
    return {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(train_dset),
        'train_dset_unaugmented': wrapped_dataset(train_dset_unaug),
        'test_loader': test_loader,
        'num_train_samples': len(train_dset),
        'classes': fashion_classes,
        'template': fashion_templates
    }
