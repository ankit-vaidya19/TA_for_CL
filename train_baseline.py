
import argparse
from vit_baseline import ViT_LoRA
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, default='oxfordpets')
parser.add_argument('-ddir', '--data-dir', type=str, default='../data')
parser.add_argument('-bs','--batch_size',type = int,default = 32)
parser.add_argument('-nw','--num_workers',type=int,default=2)
parser.add_argument('-nc','--num_classes',type = int,default=10)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)

parser.add_argument('--lora-r', type=int, default=16, help="The dimension used by the LoRA update matrices")
parser.add_argument('--lora-alpha', type=int, default=16, help="scaling factor")
parser.add_argument('--lora-dropout', type=float, default=0.1)
parser.add_argument('--lora-bias', type=str, default="none", help="if bias params should be trained or not")

args = parser.parse_args()

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")



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
                transforms.RandomResizedCrop(processor.size["height"]),
                transforms.RandomHorizontalFlip(),
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
                transforms.CenterCrop(processor.size["height"]),
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

model = ViT_LoRA(args , use_LoRA=True)
model.fit(args, train_loader, test_loader)
model.test(test_loader)
