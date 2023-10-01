
import argparse
from vit_baseline import ViT_LoRA
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor
from utils import Logger

import os, sys


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, default='oxfordpets')
parser.add_argument('-ddir', '--data-dir', type=str, default='../data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
parser.add_argument('-bs','--batch_size',type = int,default = 32)
parser.add_argument('-nw','--num_workers',type=int,default=2)
parser.add_argument('-nc','--num_classes',type = int,default=10)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('--device', type=str, default="cuda:0")

parser.add_argument('-lora','--use-lora', type=bool, default=None)

parser.add_argument('--lora-r', type=int, default=16, help="The dimension used by the LoRA update matrices")
parser.add_argument('--lora-alpha', type=int, default=16, help="scaling factor")
parser.add_argument('--lora-dropout', type=float, default=0.1)
parser.add_argument('--lora-bias', type=str, default="none", help="if bias params should be trained or not")

parser.add_argument('--tasknum', type=int, default=None)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

sys.stdout = Logger(os.path.join(args.output_dir, 'logs-task-{}.txt'.format(args.data)))

print(args)

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

if args.tasknum == None:
    args.tasknum = args.data + '_baseline'

if args.data == "oxfordpets":
    train_ds = datasets.OxfordIIITPet(
        root=args.data_dir,
        split="trainval",
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(processor.size["height"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        ),
    )

    test_ds = datasets.OxfordIIITPet(
        root=args.data_dir,
        split="test",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(processor.size["height"]),
                transforms.CenterCrop(processor.size["height"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        ),
    )
elif args.data == "svhn":
    train_ds = datasets.SVHN(
        root=args.data_dir,
        split="train",
        download=True,
        transform=transforms.Compose(
            [    
                transforms.Resize(processor.size["height"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        ),
    )

    test_ds = datasets.SVHN(
        root=args.data_dir,
        split="test",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(processor.size["height"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        ),
    )
elif args.data == "oxfordflowers":
    train_ds = datasets.Flowers102(
        root=args.data_dir,
        split="train",
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(processor.size["height"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        ),
    )

    test_ds = datasets.Flowers102(
        root=args.data_dir,
        split="test",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(processor.size["height"]),
                transforms.CenterCrop(processor.size["height"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            ]
        ),
    )

train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)
test_loader = DataLoader(
    test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)

if args.use_lora:
    print("\nINFO: Using LoRA")
    args.use_lora = True
else:
    print("\nINFO: NOT using LoRA")
    args.use_lora = False
    
model = ViT_LoRA(args , use_LoRA=args.use_lora)
model.fit(args, train_loader, test_loader)
model.test(test_loader)
