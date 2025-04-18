### [Task Arithmetic with LoRA for Continual Learning](https://arxiv.org/abs/2311.02428)
**Abstract:** <br>
Continual learning refers to the problem where the training data is available in sequential chunks, termed "tasks". The majority of progress in continual learning has been stunted by the problem of catastrophic forgetting, which is caused by sequential training of the model on streams of data. Moreover, it becomes computationally expensive to sequentially train large models multiple times. To mitigate both of these problems at once, we propose a novel method to continually train transformer-based vision models using low-rank adaptation and task arithmetic. Our method completely bypasses the problem of catastrophic forgetting, as well as reducing the computational requirement for training models on each task. When aided with a small memory of 10 samples per class, our method achieves performance close to full-set finetuning. We present rigorous ablations to support the prowess of our method.

#### Repository Structure:-
* `data.py` : _Dataloaders_ for different datasets we experimented on.
* `vit_baseline.py` : Initializes the _LoRA-augmented ViT architecture_ and defines the model training and testing functions. 
* `train_baseline.py` : Script for training ViT in an _offline setting_.
* `train.py` : Script for training ViT in a _continual setting_.
* `test_model.py` : Alternate script for testing the trained network. 
* `task_vectors.py` : Task arithmetic methods.
* `apply_ta.py` : Methods to apply task arithmetic to the trained ViT.
* `evaluate.py` : Generates, fine-tunes and evaluates the Task-agnostic Vector.
* `utils.py` : Holds Logger utils.

