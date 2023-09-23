import torch
import numpy as np

from data import TaskDataset
from apply_ta import get_model

from vit_baseline import ViT_LoRA
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor
import torch
from utils import Logger
import sys, os, time

from data import TaskDataset

import argparse

parser = argparse.ArgumentParser()


parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('-nw','--num-workers', type=int, default=2)
parser.add_argument('--test-interval', type=int, default=1)
# parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--lora-r', type=int, default=16, help="The dimension used by the LoRA update matrices")
parser.add_argument('--lora-alpha', type=int, default=16, help="scaling factor")
parser.add_argument('--lora-dropout', type=float, default=0.1)
parser.add_argument('--lora-bias', type=str, default="none", help="if bias params should be trained or not")

parser.add_argument('-d', '--data', type=str, default='oxfordpet')
parser.add_argument('-ddir', '--data-dir', type=str, default='../data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-t', '--tasknum', type=int)
args = parser.parse_args()

print(args)


img_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

test_all_tasks = list()

for task_idx in range(6):
    args.tasknum = task_idx
    trainset, testset = TaskDataset(args, img_processor).get_datasets()
    # trainloader = torch.utils.data.DataLoader(
    #         trainset, batch_size=16, shuffle=True,
    #         num_workers=args.num_workers
    #     )
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=16, shuffle=False,
            num_workers=args.num_workers
        )
    print(f"Length of {task_idx}th test dataset", len(testset))
    test_all_tasks.append(testloader)

final_model = get_model(args, "/content/vit_pretrained.pt", list_of_task_checkpoints=[f"/content/vit_task_{i}_best.pt" for i in range(6)], scaling_coef=0.25)
# print(final_model)
for task_idx, loader in enumerate(test_all_tasks):
    print(task_idx)
    final_model.test(loader)