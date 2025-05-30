from vit_baseline import ViT_LoRA
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor
import torch
from utils import Logger
import sys, os, time

from data import TaskDataset, task_dict

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('-p', '--patience', type=int, default=10)
parser.add_argument('-nw','--num-workers', type=int, default=2)
parser.add_argument('--test-interval', type=int, default=1)
parser.add_argument('--device', type=str, default="cuda:0")
# parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--lora-r', type=int, default=16, help="The dimension used by the LoRA update matrices")
parser.add_argument('--lora-alpha', type=int, default=16, help="scaling factor")
parser.add_argument('--lora-dropout', type=float, default=0.1)
parser.add_argument('--lora-bias', type=str, default="none", help="if bias params should be trained or not")

parser.add_argument('-expt', '--expt-type', type=str, default=None) # TODO : can also be "FShotTuning"

parser.add_argument('-d', '--data', type=str, default='oxfordpet')
parser.add_argument('-ddir', '--data-dir', type=str, default='./data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-t', '--tasknum', type=int)
args = parser.parse_args()

parser.add_argument('-nc', '--num-classes', type=int, default=None)
args = parser.parse_args()

if args.num_classes == None:
    if args.data == 'oxfordpet':
        args.num_classes = 37
    elif args.data == 'svhn' or args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'oxfordflowers':
        args.num_classes = 102
    elif args.data == 'stanfordcars':
        args.num_classes = 196


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

start_time = time.time()

model = ViT_LoRA(args, use_LoRA=True)
if args.expt_type == 'KLDivLoss':
    model_pretrained = ViT_LoRA(args, use_LoRA=False)
    model.fit(args, trainloader, testloader, model_pretrained)
else:
    model.fit(args, trainloader, testloader)

end_time = time.time()

print("Time taken for training : ", round(end_time-start_time, 3), "secs")



