from torchvision import datasets, transforms
import torch
import numpy as np
import random

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
mnist_templates = [
    'a black and white photo of the number {}.',
    'a blurry photo of the number {}',
    'a photo of a hand-written {}',
    'a high contrast photo of the number {}'
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
    
    dst_train_unaugmented = datasets.MNIST(
        config['dataset']['root'], train=True, download=True, transform= test_transform)
    
    dst_test = datasets.MNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)
    # class_names = dst_train.classes
    # dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    # dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']
    test_loader = torch.utils.data.DataLoader(
        wrapped_dataset(dst_test), batch_size = config['training_opt']['test_batch_size'],
        shuffle=False, num_workers = config['training_opt']['num_data_workers'], pin_memory=True, drop_last=False
    )

    return {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train),
        'train_dset_unaugmented': wrapped_dataset(dst_train_unaugmented),
        'test_loader': test_loader,
        'num_train_samples': len(dst_train),
        'classes': mnist_classes,
        'template': mnist_templates
    }

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
    
    dst_train_unaugmented = datasets.FashionMNIST(
        config['dataset']['root'], train=True, download=True, transform= test_transform)
    
    dst_test = datasets.FashionMNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)
    # class_names = dst_train.classes
    # dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    # dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']
    test_loader = torch.utils.data.DataLoader(
        wrapped_dataset(dst_test), batch_size = config['training_opt']['test_batch_size'],
        shuffle=False, num_workers = config['training_opt']['num_data_workers'], pin_memory=True, drop_last=False
    )

    return {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train),
        'train_dset_unaugmented': wrapped_dataset(dst_train_unaugmented),
        'test_loader': test_loader,
        'num_train_samples': len(dst_train),
    }