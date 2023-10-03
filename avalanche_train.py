import torch
import torch.nn as nn
# from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!
from avalanche.logging import TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics

from avalanche_data import CustomDataset, ViT_basic


def accuracy(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    acc = np.sum((true == pred).astype(np.float32)) / len(true)
    return acc * 100

def train():
    pass

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

parser.add_argument('--strategy', type=str, default="replay")

parser.add_argument('-d', '--data', type=str, default='oxfordpet')
parser.add_argument('-ddir', '--data-dir', type=str, default='./data')
parser.add_argument('-odir', '--output-dir', type=str, default='./output')
args = parser.parse_args()

parser.add_argument('-nc', '--num-classes', type=int, default=None)
args = parser.parse_args()

if args.num_classes == None:
    if args.data == 'oxfordpet':
        args.num_classes = 37
    elif args.data == 'svhn':
        args.num_classes = 10
    elif args.data == 'oxfordflowers':
        args.num_classes = 102
    elif args.data == 'stanfordcars':
        args.num_classes = 196

print(args)



model = ViT_basic(args)
scenario = CustomDataset(args).get_scenario()

text_logger = TextLogger(open('log_{args.data}_{args.strategy}.txt', 'a'))

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=args.num_classes, save_image=False),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[text_logger]
)

optimizer = torch.optim.Adam(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
criterion = nn.CrossEntropyLoss()


cl_strategy = Naive(
    model, optimizer, criterion,
    train_mb_size=100, train_epochs=1, eval_mb_size=100, evaluator=eval_plugin
)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    res = cl_strategy.train(experience, num_workers = 2)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(scenario.test_stream))