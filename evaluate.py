import torch
import numpy as np

from data import TaskDataset, task_dict, FewShotDataset
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

parser.add_argument('-nspc', '--n-samples', type=int, default=None)
parser.add_argument('-e', '--epochs', type=int, default=10)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('-nw','--num-workers', type=int, default=2)
parser.add_argument('--test-interval', type=int, default=1)
parser.add_argument('--device', type=str, default="cuda:0")
# parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--lora-r', type=int, default=16, help="The dimension used by the LoRA update matrices")
parser.add_argument('--lora-alpha', type=int, default=16, help="scaling factor")
parser.add_argument('--lora-dropout', type=float, default=0.1)
parser.add_argument('--lora-bias', type=str, default="none", help="if bias params should be trained or not")

parser.add_argument('-d', '--data', type=str, default='oxfordpet')
parser.add_argument('-ddir', '--data-dir', type=str, default='../data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-midir', '--model-input-dir', type=str, default='./data')
parser.add_argument('-t', '--tasknum', type=int)
parser.add_argument('-tot', '--total-tasks', type=int)
parser.add_argument('-scoef', '--scaling-coef', type=float, default=0.25)

args = parser.parse_args()

parser.add_argument('-nc', '--num-classes', type=int, default=None)
args = parser.parse_args()

if args.num_classes == None:
    args.num_classes = 10 if args.data=='svhn' else( 37 if args.data=='oxfordpet' else 102)



if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)
    
sys.stdout = Logger(os.path.join(args.output_dir, 'logs-evaluate-{}.txt'.format(args.data)))

print(args)

img_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")



# get pretrained model
pretrained_model = ViT_LoRA(args, use_LoRA=True)
torch.save(pretrained_model, f"{args.output_dir}/vit_pretrained.pt")

# get final model
final_model = get_model(args, f"{args.output_dir}/vit_pretrained.pt", list_of_task_checkpoints=[f"{args.model_input_dir}/vit_task_{i}_best.pt" for i in range(args.total_tasks)], scaling_coef=args.scaling_coef, return_trainable=True)

if args.n_samples: # run only if want to train on n_samples of each class
    fewshot_trainset, alltask_testset = FewShotDataset(args.data, args.data_dir, img_processor, args.n_samples).get_datasets()

    fewshot_trainloader = torch.utils.data.DataLoader(
            fewshot_trainset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers
        )
    
    alltask_testloader = torch.utils.data.DataLoader(
            alltask_testset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers
        )

    # finetune model on n_samples_per_class
    print("\nINFO: Training FINAL_MODEL on {args.n_samples} samples of each class")
    final_model.fit(args, fewshot_trainloader)

    # test the model on entire dataset
    print("\nINFO: Testing on entire testset (all tasks)")
    final_model.test(alltask_testloader)

# test the final model on task-wise testsets
test_all_tasks = list()

for task_idx in range(args.total_tasks):
    args.tasknum = task_idx
    trainset, testset = TaskDataset(args, img_processor).get_datasets()
    # trainloader = torch.utils.data.DataLoader(
    #         trainset, batch_size=16, shuffle=True,
    #         num_workers=args.num_workers
    #     )
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers
        )
    print(f"Length of {task_idx}th test dataset", len(testset))
    test_all_tasks.append(testloader)

# print(final_model)
for task_idx, loader in enumerate(test_all_tasks):
    print(f"\nINFO: Testing on testset of TASK {task_idx}") 
    final_model.test(loader)

torch.save(final_model, f"{args.output_dir}/resultant_model_{args.data}.pt")
