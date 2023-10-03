from torchvision import datasets, transforms
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
import torch
from transformers import ViTModel
import torch.nn as nn


class ViT_basic(nn.Module):
    def __init__(self, args, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.device = args.device
        self.model_name = model_name
        
        self.ViT = ViTModel.from_pretrained(self.model_name)
        self.linear = nn.Linear(768, args.num_classes)
        self.print_trainable_parameters()
        self.ViT.to(self.device)
        self.linear.to(self.device)

    def forward(self, x):
        x = self.ViT(x).pooler_output
        return self.linear(x)

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.ViT.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

class CustomDataset():
    def __init__(self, args):
        self.args = args
        self.trainset = datasets.OxfordIIITPet(root=args.data_dir, split='trainval', download=True, transform=transforms.ToTensor())
        self.testset = datasets.OxfordIIITPet(root=args.data_dir, split='test', download=True, transform=transforms.ToTensor())
        self.trainset.targets = self.trainset._labels
        self.testset.targets = self.testset._labels

        self.task_dict = {0:6, 1:6, 2:6, 3:6, 4:6, 5:7}
    
    def get_scenario(self):
        scenario = nc_benchmark(
            self.trainset, self.testset, n_experiences=6, per_exp_classes=self.task_dict, shuffle=True, seed=1234,
            task_labels=True
        )
        print("Trainset : ", len(self.trainset), "Testset : ", len(self.testset))
        return scenario
