from task_vectors import TaskVector
import torch
from vit_baseline import ViT_LoRA


task_dict = {'oxfordpet' : {0 : ['american_bulldog', 'scottish_terrier', 'english_setter', 'newfoundland', 'Maine_Coon', 'British_Shorthair'],
                            1 : ['Persian', 'boxer', 'english_cocker_spaniel', 'saint_bernard', 'Russian_Blue', 'Bombay'],
                            2 : ['japanese_chin', 'Sphynx', 'german_shorthaired', 'basset_hound', 'samoyed', 'shiba_inu'],
                            3 : ['staffordshire_bull_terrier', 'Siamese', 'wheaten_terrier', 'Abyssinian', 'keeshond', 'havanese'],
                            4 : ['yorkshire_terrier', 'Bengal', 'great_pyrenees', 'Egyptian_Mau', 'pomeranian', 'beagle'],
                            5 : ['american_pit_bull_terrier', 'Ragdoll', 'miniature_pinscher', 'pug', 'Birman', 'leonberger', 'chihuahua']},
            
            'svhn' : {  0 : ['0', '8'],
                        1 : ['1', '7'],
                        2 : ['2', '5'],
                        3 : ['3', '6'],
                        4 : ['4', '9']
                        },

            'oxfordflowers' : { 0 : [x for x in range(0,10)] , 1 : [x for x in range(10, 20)],
                                2 : [x for x in range(20,30)], 3 : [x for x in range(30, 40)],
                                4 : [x for x in range(40,50)], 5 : [x for x in range(50, 60)],
                                6 : [x for x in range(60,70)], 7 : [x for x in range(70, 80)],
                                8 : [x for x in range(80,90)], 9 : [x for x in range(90, 102)]
                            }  
            
            }

# for i in range(6):
#     print(len(task_dict["oxfordpet"][i]))
# raise
ranges_of_classes = {
    "oxfordpet": {
        0: (0, 6),
        1: (6, 12),
        2: (12, 18),
        3: (18, 24),
        4: (24, 30),
        5: (30, 37)
    },

    "svhn": {
        0: (0, 2),
        1: (2, 4),
        2: (4, 6),
        3: (6, 8),
        4: (8, 10),
    },

    "oxfordflowers" : { 
        0: (0, 10), 1: (10, 20),
        2: (20, 30), 3: (30, 40),
        4: (40, 50), 5: (50, 60),
        6: (60, 70), 7: (70, 80),
        8: (80, 90), 9: (90, 102),                        
    }

}


def get_model(args, pretrained_checkpoint, list_of_task_checkpoints, scaling_coef):
    task_vector_list = [
        TaskVector(pretrained_checkpoint, task_checkpoint)
        for task_checkpoint in list_of_task_checkpoints
    ]
    vector_sum = sum(task_vector_list)
    model = vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    
    for param in model.parameters():
        param.requires_grad = False
    for task_idx, ckpt in enumerate(sorted(list_of_task_checkpoints)):
        finetuned_weights = torch.load(ckpt)
        # task_model = ViT_LoRA(args, use_)
        
        taskwise_model = ViT_LoRA(args, use_LoRA=True)
        taskwise_model.load_state_dict(finetuned_weights)
        # print(taskwise_model.linear.weight.shape)
        start_idx = ranges_of_classes[args.data][task_idx][0]
        end_idx = ranges_of_classes[args.data][task_idx][1]
        model.linear.weight[start_idx:end_idx , :] = taskwise_model.linear.weight[start_idx:end_idx , :]
        model.linear.bias[start_idx:end_idx] = taskwise_model.linear.bias[start_idx:end_idx]
        # print(model.linear.weight.shape)
        # print(model.linear.bias.shape)
    return model
