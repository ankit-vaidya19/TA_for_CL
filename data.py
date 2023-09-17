import torch
# import numpy as np
# import torchvision
from torchvision import datasets, transforms

def get_dataloaders(args: str):
    if args.data == 'oxfordpet':
        trainset = datasets.OxfordIIITPet(root='data_dir', 
                            split='trainval', download=True, 
                            transform=transforms.ToTensor())
        testset = datasets.OxfordIIITPet(root='data_dir', 
                            split='test', download=True, 
                            transform=transforms.ToTensor())

    elif args.data == 'stanfordcars': # wont' work as it's down for a long time now
        # CHECK
        # https://github.com/pytorch/vision/issues/7545#issuecomment-1575410733
        trainset = datasets.StanfordCars(root='data_dir', 
                            split='train', download=True, 
                            transform=transforms.ToTensor())
        testset = datasets.StanfordCars(root='data_dir', 
                            split='test', download=True, 
                            transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    return trainloader, testloader