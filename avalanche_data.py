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
        self.linear = nn.Linear(768, 10)
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
    def __init__(self, args, processor):
        self.args = args
        self.train_transform =  transforms.Compose([
                                transforms.RandomResizedCrop(processor.size["height"]),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
                                ]) 
        self.test_transform = transforms.Compose([transforms.Resize(processor.size["height"]),
                                transforms.CenterCrop(processor.size["height"]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
                                ])
        self.set_data_variables()

    def set_data_variables(self):
        if self.args.data == 'oxfordpet':
            self.trainset = datasets.OxfordIIITPet(root=self.args.data_dir, split='trainval', download=True, transform=self.train_transform)
            self.testset = datasets.OxfordIIITPet(root=self.args.data_dir, split='test', download=True, transform=self.test_transform)
            self.trainset.targets = self.trainset._labels
            self.testset.targets = self.testset._labels
            self.task_dict = {0:6, 1:6, 2:6, 3:6, 4:6, 5:7}
        
        elif self.args.data == 'oxfordflowers':
            self.trainset = datasets.Flowers102(root=self.args.data_dir, split='train', download=True, transform=self.train_transform)
            self.testset = datasets.Flowers102(root=self.args.data_dir, split='test', download=True, transform=self.test_transform)
            self.trainset.targets = self.trainset._labels
            self.testset.targets = self.testset._labels
            self.task_dict = {0:10, 1:10, 2:10, 3:10, 4:10, 5:10, 6:10, 7:10, 8:10, 9:12}
        
        elif self.args.data == 'stanfordcars':
            self.task_dict = {0:19, 1:19, 2:19, 3:19, 4:20, 5:20, 6:20, 7:20, 8:20, 9:20}
            pass

        elif self.args.data == 'cifar10':
            self.trainset = datasets.CIFAR10(root='oxfordpet', train=True, download=True, transform=self.train_transform)
            self.testset = datasets.CIFAR10(root='oxfordpet', train=False, download=True, transform=self.test_transform)
            self.task_dict = {0:2, 1:2, 2:2, 3:2, 4:2}
            
        else:
            ValueError()
            
    def get_scenario(self):
        scenario = nc_benchmark(
            self.trainset, self.testset, n_experiences=len(self.task_dict), per_exp_classes=self.task_dict, shuffle=True, seed=1234,
            task_labels=True
        )
        print("Trainset : ", len(self.trainset), "Testset : ", len(self.testset))
        return scenario
