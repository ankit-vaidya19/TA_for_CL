import argparse
from vit_baseline import ViT_LoRA
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, default='oxfordpet')
parser.add_argument('-ddir', '--data-dir', type=str, default='../data')
parser.add_argument('-bs','--batch_size',type = int,default = 32)
parser.add_argument('-nw','-num_workers',type=int,default=2)
parser.add_argument('-nc','num_classes',type = int,default=10)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch-size', type=int, default=16)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)

args = parser.parse_args()

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

train_ds = datasets.args.data(
    root=args.ddrir,
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

test_ds = datasets.OxfordIIITPet(
    root=args.ddir,
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

model = ViT_LoRA(args , use_LoRA=True)
model.fit(train_loader=train_loader)
model.test(test_loader=test_loader)