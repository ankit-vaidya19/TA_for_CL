import torch
import torch.nn as nn
from transformers import ViTModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
# from ml_collections import ConfigDict


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# cfg = ConfigDict
# cfg.epochs = 50
# cfg.batch_size = 16
# cfg.lr = 5e-6
# cfg.weight_decay = 1e-6


class ViT_LoRA(nn.Module):
    def __init__(self, args, model_name="google/vit-base-patch16-224", use_LoRA=False):
        super().__init__()
        self.model_name = model_name
        self.use_LoRA = use_LoRA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        if self.use_LoRA:
            self.ViT = ViTModel.from_pretrained(self.model_name)
            self.linear = nn.Linear(768, 37)
            self.config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["pooler"],
            )
            self.ViT = get_peft_model(self.ViT, self.config)
            self.print_trainable_parameters()
        else:
            self.ViT = ViTModel.from_pretrained(self.model_name)
            self.linear = nn.Linear(768, 37)
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

    def accuracy(self, true, pred):
        true = np.array(true)
        pred = np.array(pred)
        acc = np.sum((true == pred).astype(np.float32)) / len(true)
        return acc * 100

    def fit(self, args, train_loader):
        optim = torch.optim.Adam(
            params=self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(args.epochs):
            print(f"{epoch}/{args.epochs}")
            train_loss = []
            train_preds = []
            train_labels = []
            for batch in tqdm(train_loader):
                imgs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                scores = self(imgs)
                loss = criterion(scores, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss.append(loss.detach().cpu().numpy())
                train_labels.append(batch[1])
                train_preds.append(scores.argmax(dim=-1))
            print(f"Train Loss - {sum(train_loss)/len(train_loss)}")
            print(
                f"Train Accuracy - {self.accuracy(torch.concat(train_labels, dim=0).cpu(),torch.concat(train_preds, dim=0).cpu())}"
            )

    def test(self, test_loader):
        criterion = nn.CrossEntropyLoss()
        self.eval()
        with torch.no_grad():
            test_loss = []
            test_preds = []
            test_labels = []
            for batch in tqdm(test_loader):
                imgs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                scores = self(imgs)
                loss = criterion(scores, labels)
                test_loss.append(loss.detach().cpu().numpy())
                test_labels.append(batch[1])
                test_preds.append(scores.argmax(dim=-1))
            print(f"Test Loss - {sum(test_loss)/len(test_loss)}")
            print(
                f"Test Accuracy - {self.accuracy(torch.concat(test_labels, dim=0).cpu(),torch.concat(test_preds, dim=0).cpu(),)}"
            )
            torch.save(self.state_dict(), "vit_basline_oxp.pt")
