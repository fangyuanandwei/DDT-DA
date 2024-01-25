from torchvision import datasets, transforms
import torch
import numpy as np
from torchvision import transforms
import os
from PIL import Image, ImageOps
from Data.Datasets_index import ImageFolderIndex
from collections import Counter
from torch.utils.data import WeightedRandomSampler


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):
    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


def load_training(root_path, dir, batch_size, kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize
         ])
    file_path = os.path.join(root_path, dir)
    data = datasets.ImageFolder(root=file_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing(root_path, dir, batch_size, kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    start_center = (256 - 224 - 1) / 2
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         PlaceCrop(224, start_center, start_center),
         transforms.ToTensor(),
         normalize
         ])
    file_path = os.path.join(root_path, dir)
    data = datasets.ImageFolder(root=file_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader


'''Index'''


def load_training_index(root_path, dir, batch_size, balance, kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize
         ])
    file_path = os.path.join(root_path, dir)
    data = ImageFolderIndex(root=file_path, transform=transform)

    if balance:
        freq = Counter(data.targets)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in data.targets]
        sampler = WeightedRandomSampler(source_weights, len(data.targets))
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=sampler, drop_last=True,
                                                   **kwargs)
        return train_loader
    else:
        train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return train_loader


def load_testing_index(root_path, dir, batch_size, kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    start_center = (256 - 224 - 1) / 2
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         PlaceCrop(224, start_center, start_center),
         transforms.ToTensor(),
         normalize
         ])
    file_path = os.path.join(root_path, dir)
    data = ImageFolderIndex(root=file_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return test_loader


def load_single(root_path, dir, batch_size, kwargs):
    start_center = (256 - 224 - 1) / 2
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         PlaceCrop(224, start_center, start_center),
         transforms.ToTensor()])
    file_path = os.path.join(root_path, dir)
    data = datasets.ImageFolder(root=file_path, transform=transform)
    return data


def load_data(args, kwargs):
    source_dir = args.source
    targte_dir = args.target

    source_train_loader = load_training(args.root_path, source_dir, args.batch_size, kwargs)
    target_train_loader = load_training(args.root_path, targte_dir, args.batch_size, kwargs)

    source_test_loader = load_testing(args.root_path, source_dir, args.batch_size, kwargs)
    target_test_loader = load_testing(args.root_path, targte_dir, args.batch_size, kwargs)

    source_signal = load_single(args.root_path, source_dir, args.batch_size, kwargs)
    target_signal = load_single(args.root_path, targte_dir, args.batch_size, kwargs)

    return source_train_loader, target_train_loader, source_test_loader, target_test_loader


def load_data_ImageFolder(args, kwargs):
    source_dir = args.source
    targte_dir = args.target

    source_train_loader = load_training(args.root_path, source_dir, args.batch_size, kwargs)
    target_train_loader = load_training(args.root_path, targte_dir, args.batch_size, kwargs)

    source_test_loader = load_testing(args.root_path, source_dir, args.batch_size, kwargs)
    target_test_loader = load_testing(args.root_path, targte_dir, args.batch_size, kwargs)

    source_signal = load_single(args.root_path, source_dir, args.batch_size, kwargs)
    target_signal = load_single(args.root_path, targte_dir, args.batch_size, kwargs)

    return [source_train_loader, target_train_loader], [source_test_loader, target_test_loader], [source_signal,
                                                                                                  target_signal]


def load_data_ImageFolder_Index(args, kwargs):
    source_dir = args.source
    targte_dir = args.target

    source_train_loader = load_training_index(args.root_path, source_dir, args.batch_size, True, kwargs)
    target_train_loader = load_training_index(args.root_path, targte_dir, args.batch_size, False, kwargs)

    source_test_loader = load_testing_index(args.root_path, source_dir, args.batch_size, kwargs)
    target_test_loader = load_testing_index(args.root_path, targte_dir, args.batch_size, kwargs)

    source_signal = load_single(args.root_path, source_dir, args.batch_size, kwargs)
    target_signal = load_single(args.root_path, targte_dir, args.batch_size, kwargs)

    return [source_train_loader, target_train_loader], [source_test_loader, target_test_loader], [source_signal,
                                                                                                  target_signal]

