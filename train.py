from vit_baseline import ViT_LoRA, cfg
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

import argparse

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")


train_ds = datasets.OxfordIIITPet(
    root="/mnt/d/TA_LORA",
    split="trainval",
    target_types="category",
    download=False,
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
    root="/mnt/d/TA_LORA",
    split="test",
    target_types="category",
    download=False,
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
    train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2
)

model = ViT_LoRA(use_LoRA=True)
model.fit(train_loader=train_loader)
# model.test(test_loader=test_loader)

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--tasknum', type=int)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-d', '--data', type=str, default='oxfordpet')
parser.add_argument('-ddir', '--data-dir', type=str)
parser.add_argument('-t', '--tasknum', type=int)
# to be continued...


