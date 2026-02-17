from torchvision import datasets, transforms
import torch
import numpy as np

class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = dataset.labels if hasattr(dataset, 'labels') else dataset.targets
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        return {
            'input': self.dataset[index][0],
            'target': int(self.dataset[index][1]),
            'index': index
        }

def SVHN(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    # SVHN mean and std
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]

    transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    use_fiftyone_dataloader = False if 'visualization' not in config else config['visualization'].get('use_fiftyone_dataloader', False)
    
    if not use_fiftyone_dataloader:
        dst_train = datasets.SVHN(config['dataset']['root'], split='train', download=True, transform=transform)
        dst_test = datasets.SVHN(config['dataset']['root'], split='test', download=True, transform=test_transform)
    else:
        # FiftyOne Zoo support for SVHN might differ slightly in naming, but standard is "svhn"
        import fiftyone.zoo as foz
        # Note: FiftyOne might load 'train' and 'test' splits differently
        dst_train = foz.load_zoo_dataset("svhn", split="train", download_if_necessary=True)
        dst_test = foz.load_zoo_dataset("svhn", split="test", download_if_necessary=True)

    config['training_opt']['test_batch_size'] = (
        config['training_opt']['batch_size'] 
        if 'test_batch_size' not in config['training_opt'] 
        else config['training_opt']['test_batch_size']
    )
    
    test_loader = torch.utils.data.DataLoader(
        wrapped_dataset(dst_test), 
        batch_size=config['training_opt']['test_batch_size'],
        shuffle=False, num_workers=config['training_opt']['num_data_workers'], 
        pin_memory=True, drop_last=False
    )

    return {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train),
        'test_loader': test_loader,
        'num_train_samples': len(dst_train)
    }
