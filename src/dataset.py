import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, random_split

def get_datasets(batch_size, data_name, val_ratio=0.2):
    if data_name == "MNIST":
        full_train_dataset = torchvision.datasets.MNIST(root="./data",
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

        test_dataset = torchvision.datasets.MNIST(root="./data",
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
    
    elif data_name == "CIFAR10":
        root = './data/CIFAR_10'
        full_train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True,\
            transform=transforms.Compose([\
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True,\
            transform=transforms.Compose([\
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    
    elif data_name == "CIFAR100":
        root = './data/CIFAR_100'
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        full_train_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True,\
            transform=transforms.Compose([\
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                    ]))

        test_dataset = torchvision.datasets.CIFAR100(root=root, train=False, download=True,\
            transform=transforms.Compose([\
                    transforms.ToTensor(),
                    normalize
                    ]))
    
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, train_loader, val_loader, test_loader


def split_dataset(dataset, num_clients, dirichlet_alpha):
    num_classes = 10
    class_indices = {i: [] for i in range(num_classes)}

    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    for indices in class_indices.values():
        np.random.shuffle(indices)
    
    client_indices = {i: [] for i in range(num_clients)}

    for indices in class_indices.values():
        proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        for i, idx in enumerate(split_indices):
            client_indices[i].extend(idx)
    
    subsets = [Subset(dataset, client_indices[i]) for i in range(num_clients)]
    return subsets