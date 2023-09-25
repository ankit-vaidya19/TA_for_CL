import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformers import AutoImageProcessor
from apply_ta import get_model


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, default='oxfordpet')
parser.add_argument('-ddir', '--data-dir', type=str, default='../data')
parser.add_argument('-bs','--batch_size',type = int,default = 32)
parser.add_argument('-nw','--num_workers',type=int,default=2)
parser.add_argument('-midir', '--model-input-dir', type=str, default='./data')
parser.add_argument('-tot', '--total-tasks', type=int)
parser.add_argument('-sc','--scaling_coefficient',type = float,default=0.25)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-lr', '--lr', type=float, default=5e-6)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
parser.add_argument('--lora-r', type=int, default=16, help="The dimension used by the LoRA update matrices")
parser.add_argument('--lora-alpha', type=int, default=16, help="scaling factor")
parser.add_argument('--lora-dropout', type=float, default=0.1)
parser.add_argument('--lora-bias', type=str, default="none", help="if bias params should be trained or not")
args = parser.parse_args()

def accuracy(true,pred):
    true = np.array(true)
    pred = np.array(pred)
    acc = np.sum((true == pred).astype(np.float32)) / len(true)
    return acc * 100

def test_model(model,test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
            test_loss = []
            test_preds = []
            test_labels = []
            for batch in tqdm(test_loader):
                imgs = torch.Tensor(batch[0]).to(device)
                labels = torch.Tensor(batch[1]).to(device)
                scores = model(imgs)
                loss = criterion(scores, labels)
                test_loss.append(loss.detach().cpu().numpy())
                test_labels.append(batch[1])
                test_preds.append(scores.argmax(dim=-1))
            loss = sum(test_loss)/len(test_loss)
            acc = accuracy(torch.concat(test_labels, dim=0).cpu(),torch.concat(test_preds, dim=0).cpu())
            print(f"\tTest:\tLoss - {round(loss, 3)}",'\t',f"Accuracy - {round(acc,3)}")
            torch.save(final_model, f"{args.output_dir}/resultant_model_{args.data}.pt")
            return loss, acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

if args.data == "oxfordpets":
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
elif args.data == "svhn":
    test_ds = datasets.SVHN(
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
elif args.data == "oxfordflowers":
    test_ds = datasets.Flowers102(
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

test_loader = DataLoader(test_ds,args.batch_size,num_workers=args.num_workers,shuffle=False)
final_model = get_model(args, f"{args.output_dir}/vit_pretrained.pt", list_of_task_checkpoints=[f"{args.model_input_dir}/vit_task_{i}_best.pt" for i in range(args.total_tasks)], scaling_coef=args.scaling_coefficient)

test_model(final_model,test_loader)
