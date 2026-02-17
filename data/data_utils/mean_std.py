# Per-dataset channel-wise mean and std used for normalization.
# Keys must match config['dataset']['name'] values.

mean = {
    "CIFAR10":      [0.4914, 0.4822, 0.4465],
    "CIFAR10_LT":   [0.4914, 0.4822, 0.4465],
    "CIFAR100":     [0.5071, 0.4865, 0.4409],
    "CIFAR100_LT":  [0.5071, 0.4865, 0.4409],
    "SVHN":         [0.4377, 0.4438, 0.4728],
    "MNIST":        [0.1307, 0.1307, 0.1307],
    "MNIST_LT":     [0.1307, 0.1307, 0.1307],
    "FashionMNIST": [0.2860, 0.2860, 0.2860],
    "TinyImageNet": [0.4802, 0.4481, 0.3975],
}

std = {
    "CIFAR10":      [0.2470, 0.2435, 0.2616],
    "CIFAR10_LT":   [0.2470, 0.2435, 0.2616],
    "CIFAR100":     [0.2673, 0.2564, 0.2762],
    "CIFAR100_LT":  [0.2673, 0.2564, 0.2762],
    "SVHN":         [0.1980, 0.2010, 0.1970],
    "MNIST":        [0.3081, 0.3081, 0.3081],
    "MNIST_LT":     [0.3081, 0.3081, 0.3081],
    "FashionMNIST": [0.3530, 0.3530, 0.3530],
    "TinyImageNet": [0.2770, 0.2691, 0.2822],
}
