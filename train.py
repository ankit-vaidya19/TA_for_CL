from vit_baseline import ViT_LoRA
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor
import torch
from utils import Logger
import sys, os

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
parser.add_argument('-ddir', '--data-dir', type=str, default='./data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-t', '--tasknum', type=int)
args = parser.parse_args()

print(args)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

sys.stdout = Logger(os.path.join(args.output_dir, 'logs-task-{}.txt'.format(args.tasknum)))

img_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

trainset, testset = TaskDataset(args, img_processor).get_datasets()
trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )
testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )


model = ViT_LoRA(args, use_LoRA=True)
model.fit(args, trainloader, testloader)





