from torchvision import datasets, transforms
import torch
import numpy as np
import random

from .data_utils.generate_noise import apply_or_generate_label_noise

mnist_classes = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

mnist_templates = [
    "a photo of the number {}.",
    "a handwritten digit {}.",
    "a grayscale image of the number {}.",
    "a photo of a handwritten {}.",
]

fashionmnist_classes = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

fashionmnist_templates = [
    "a photo of a {}.",
    "a grayscale photo of a {}.",
    "a photo of a person wearing a {}.",
    "a photo of a {} on a white background.",
    "a photo of a {} item.",
]

class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = dataset.targets
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return {
            'input': self.dataset[index][0],
            'target': self.dataset[index][1],
            'index': index
        }


def _build_test_loader(config, dst_test):
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']
    return torch.utils.data.DataLoader(
        wrapped_dataset(dst_test), batch_size = config['training_opt']['test_batch_size'],
        shuffle=False, num_workers = config['training_opt']['num_data_workers'], pin_memory=True, drop_last=False
    )


def _build_dataset_info(config, logger, dataset_name, dst_train, dst_test, num_classes, classes, templates, include_noise=False):
    payload = {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train),
        'test_loader': _build_test_loader(config, dst_test),
        'num_train_samples': len(dst_train),
        'classes': classes,
        'template': templates,
    }
    if include_noise:
        payload.update(
            apply_or_generate_label_noise(
                dataset=dst_train,
                num_classes=num_classes,
                dataset_config=config['dataset'],
                logger=logger,
                dataset_name=dataset_name,
                seed=config.get('seed'),
            )
        )
        payload['train_dset'] = wrapped_dataset(dst_train)
    return payload
    
def MNIST(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 28 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 28 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    dst_train = datasets.MNIST(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    
    
    dst_test = datasets.MNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)
    # class_names = dst_train.classes
    # dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    # dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='MNIST',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=mnist_classes,
        templates=mnist_templates,
    )


def MNIST_Noise(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 28 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 28 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    dst_train = datasets.MNIST(
        config['dataset']['root'], train=True, download=True, transform= transform
    )

    dst_test = datasets.MNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)

    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='MNIST',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=mnist_classes,
        templates=mnist_templates,
        include_noise=True,
    )

def FashionMNIST(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.2860]
    std = [0.3530]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 28 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 28 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    dst_train = datasets.FashionMNIST(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    
    
    dst_test = datasets.FashionMNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)
    # class_names = dst_train.classes
    # dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    # dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='FashionMNIST',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=fashionmnist_classes,
        templates=fashionmnist_templates,
    )


def FashionMNIST_Noise(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.2860]
    std = [0.3530]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 28 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 28 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    dst_train = datasets.FashionMNIST(
        config['dataset']['root'], train=True, download=True, transform= transform
    )

    dst_test = datasets.FashionMNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)

    return _build_dataset_info(
        config=config,
        logger=logger,
        dataset_name='FashionMNIST',
        dst_train=dst_train,
        dst_test=dst_test,
        num_classes=num_classes,
        classes=fashionmnist_classes,
        templates=fashionmnist_templates,
        include_noise=True,
    )