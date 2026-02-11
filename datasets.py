from torchvision import datasets, transforms
import math
import random

import torch
from torch.utils.data.dataset import Subset


def build_cifar_transform(is_train, args):
    resize_im = args.input_size > 32

    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation = 3),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness = 63 / 225),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5071, 0.4867, 0.4408), std = (0.2675, 0.2565, 0.2761)),
        ])
        return transform
    
    transform = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        transform.append(transforms.Resize(size, interpolation = 3)) # to maintain same ratio w.r.t. 224 images
        transform.append(transforms.CenterCrop(args.input_size))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean = (0.5071, 0.4867, 0.4408), std = (0.2675, 0.2565, 0.2761)))

    return transforms.Compose(transform)


def build_transform(is_train, args):
    resize_im = args.input_size > 32

    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale = scale, ratio = ratio),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor(),
        ])
        return transform
    
    transform = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        transform.append(transforms.Resize(size, interpolation = 3)) # to maintain smae ratio w.r.t 224 images
        transform.append(transforms.CenterCrop(args.input_size))
    transform.append(transforms.ToTensor())

    return transforms.Compose(transform)


def get_dataset(dataset, transform_train, transform_val, args, target_transform = None):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train = True, download = True, transform = transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train = False, download = True, transform = transform_val)
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val


def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    classes_per_task = math.ceil(nb_classes / args.num_tasks)

    labels = [i for i in range(nb_classes)]
    if args.shuffle:
        random.shuffle(labels)

    split_datasets = list()
    mask = list()
    target_task_map = {}

    for i in range(args.num_tasks):

        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)
        for k in scope:
            target_task_map[k] = i
        
        train_split_indices = []
        test_split_indices = []

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
        
        for k in range(len(dataset_val.targets)):
            if int(dataset_val.targets[k]) in scope:
                test_split_indices.append(k)
        
        subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])

    return split_datasets, mask, target_task_map


def split_single_class_dataset(dataset_train, dataset_val, mask, args):
    nb_classes = len(dataset_val.classes)
    split_datasets = dict()

    for i in range(len(mask)):
        single_task_labels = mask[i]

        for cls_id in single_task_labels:
            train_split_indices = []
            test_split_indices = []

            for k in range(len(dataset_train.targets)):
                if dataset_train.target_transform is not None:
                    if int(dataset_train.target_transform(dataset_train.targets[k])) == cls_id:
                        train_split_indices.append(k)
                elif int(dataset_train.targets[k]) == cls_id:
                    train_split_indices.append(k)
            
            for k in range(len(dataset_val.targets)):
                if dataset_val.target_transform is not None:
                    if int(dataset_val.target_transform(dataset_val.targets[k])) == cls_id:
                        test_split_indices.append(k)
                elif int(dataset_val.targets[k]) == cls_id:
                    test_split_indices.append(k)
            
            subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

            split_datasets[cls_id] = [subset_train, subset_val]
    
    return split_datasets


class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    # init
    dataloader = list()
    dataloader_per_cls = dict()
    class_mask = list() if args.task_inc or args.train_mask else None
    target_task_map = dict()

    if 'cifar' in args.dataset.lower():
        transform_train = build_cifar_transform(True, args)
        transform_val = build_cifar_transform(False, args)
    else:
        transform_train = build_transform(True, args)
        transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-', ''), transform_train, transform_val, args)
        dataset_train_mean, dataset_val_mean = get_dataset(args.dataset.replace('Split-', ''), transform_train, transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask, target_task_map = split_single_dataset(dataset_train, dataset_val, args)
        splited_dataset_per_cls = split_single_class_dataset(dataset_train_mean, dataset_val_mean, class_mask, args)
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        else:
            dataset_list = args.dataset.split(',')

        if args.shuffle:
            random.shuffle(dataset_list)

        args.nb_classes = 0
        splited_dataset_per_cls = {}

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]
        else:
            if 'cifar' in dataset_list[i].lower():
                transform_train = build_cifar_transform(True, args)
                transform_val = build_cifar_transform(False, args)
            
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)
            dataset_train_mean, dataset_val_mean = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)
            
            if class_mask is not None and target_task_map is not None:
                class_mask.append([i + args.nb_classes for i in range(len(dataset_val.classes))])

                for j in range(len(dataset_val.classes)):
                    target_task_map[j + args.nb_classes] = i

                args.nb_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
                dataset_train_mean.target_transform = transform_target
                dataset_val_mean.target_transform = transform_target
            

            splited_dataset_per_cls.update(split_single_class_dataset(dataset_train_mean, dataset_val_mean, [class_mask[i]], args))


            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.RandomSampler(dataset_val)


            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler = sampler_train,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                pin_memory = args.pin_mem
            )

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler = sampler_val,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                pin_memory = args.pin_mem
            )

            dataloader.append({'train': data_loader_train, 'val': data_loader_val})
    
    for i in range(len(class_mask)):
        for cls_id in class_mask[i]:
            dataset_train_cls, dataset_val_cls = splited_dataset_per_cls[cls_id]

            sampler_train = torch.utils.data.RandomSampler(dataset_train_cls)
            sampler_val = torch.utils.data.RandomSampler(dataset_val_cls)


            data_loader_train_cls = torch.utils.data.DataLoader(
                dataset_train_cls, sampler = sampler_train,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                pin_memory = args.pin_mem
            )

            data_loader_val_cls = torch.utils.data.DataLoader(
                dataset_val_cls, sampler = sampler_val,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                pin_memory = args.pin_mem
            )

            dataloader_per_cls[cls_id] = {'train': data_loader_train_cls, 'val': data_loader_val_cls}
    
    return dataloader, dataloader_per_cls, class_mask, target_task_map




        

