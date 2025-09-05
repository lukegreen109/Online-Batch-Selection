from torchvision import datasets, transforms
import torch
import numpy as np

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

class IMBALANCEMNIST(datasets.MNIST):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, reverse=False):
        super(IMBALANCEMNIST, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num =  img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                else:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = torch.tensor(new_targets, dtype=torch.long)
        
    def get_cls_num_list(self):
        return [self.num_per_cls_dict[i] for i in range(self.cls_num)]


def MNIST(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = (0.1307,)
    std = (0.3081,)

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

    use_fiftyone = False if 'visualization' not in config else config['visualization']['use_fiftyone']
    if not use_fiftyone:
        dst_train = datasets.MNIST(config['dataset']['root'], train=True, download=True, transform=transform)
        dst_test = datasets.MNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)

    else:
        import fiftyone.zoo as foz
        dst_train = foz.load_zoo_dataset("mnist", split="train", download_if_necessary=True)
        dst_test = foz.load_zoo_dataset("mnist", split="test", download_if_necessary=True)

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


def MNIST_LT(config, logger):
    im_size = (28, 28) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = (0.1307,)
    std = (0.3081,)

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
    use_fiftyone = False if 'visualization' not in config else config['visualization']['use_fiftyone']
    if not use_fiftyone:
        dst_train = IMBALANCEMNIST(
            root=config['dataset']['root'], 
            imb_factor=config['dataset']['imb_factor'], 
            rand_number=config['dataset']['rand_number'], 
            train=True, download=True, transform=transform
        )
        dst_test = datasets.MNIST(config['dataset']['root'], train=False, download=True, transform=test_transform)
    else:
        import fiftyone.zoo as foz
        dst_train = foz.load_zoo_dataset("mnist", split="train", download_if_necessary=True)
        dst_test = foz.load_zoo_dataset("mnist", split="test", download_if_necessary=True)

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
