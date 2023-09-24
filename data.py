import torch
import glob
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import cv2

# How to call the loader:
"""
taskset = TaskDataset(args)
trainloader, testloader = taskset.get_datasets()
"""

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

# label2int = {'oxfordpet' : {'american_bulldog': 0, 'scottish_terrier': 1, 'english_setter': 2, 'newfoundland': 3, 'Maine_Coon': 4, 'British_Shorthair': 5,
#                              'Persian': 6, 'boxer': 7, 'english_cocker_spaniel': 8, 'saint_bernard': 9, 'Russian_Blue': 10, 'Bombay': 11, 'japanese_chin': 12,
#                              'Sphynx': 13, 'german_shorthaired': 14, 'basset_hound': 15, 'samoyed': 16, 'shiba_inu': 17, 'staffordshire_bull_terrier': 18,
#                              'Siamese': 19, 'wheaten_terrier': 20, 'Abyssinian': 21, 'keeshond': 22, 'havanese': 23, 'yorkshire_terrier': 24, 'Bengal': 25,
#                              'great_pyrenees': 26, 'Egyptian_Mau': 27, 'pomeranian': 28, 'beagle': 29, 'american_pit_bull_terrier': 30, 
#                              'Ragdoll': 31, 'miniature_pinscher': 32, 'pug': 33, 'Birman': 34, 'leonberger': 35, 'chihuahua': 36}
#             }

class ImageDataset(Dataset):
    def __init__(self, image_list, label_list, split, processor):
        super().__init__()
        self.image_list = image_list
        self.label_list = label_list

        if split == 'train':
            self.transform = transforms.Compose([
                                transforms.RandomResizedCrop(processor.size["height"]),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
                                ]) 
        elif split == 'test':
            self.transform = transforms.Compose([transforms.Resize(processor.size["height"]),
                                transforms.CenterCrop(processor.size["height"]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
                                ])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.image_list[idx]).convert('RGB'))
        label = self.label_list[idx]
        return img, label


class TaskDataset():
    def __init__(self, args, img_processor, task_dict=task_dict):
        self.args = args
        self.img_processor = img_processor
        self.task_dict = task_dict[args.data]
        self.label2int = self.get_label2int()
        self.train_imgs, self.train_labels = [],[]
        self.test_imgs, self.test_labels = [],[]
        self.get_lists()

    def get_label2int(self):
        label2int = []
        for task in self.task_dict.values():
            for clas in task:
                label2int.append(clas)

        keys = [x for x in range(len(label2int))]
        label2int = dict(zip(label2int, keys))
        return label2int
        
    def get_lists(self):
        if self.args.data == 'oxfordpet':
            oxfordpet = datasets.OxfordIIITPet(root=self.args.data_dir, download=True)

            #trainset lists
            with open(f'{self.args.data_dir}/oxford-iiit-pet/annotations/trainval.txt', 'r') as f_train:
                lines = f_train.readlines()
                # print(lines)
                for img in lines:
                    img_name = img.split(' ')[0]
                    img_label = img_name.rsplit('_', 1)[0]
                    if img_label in self.task_dict[self.args.tasknum]:
                        self.train_imgs.append(f'{self.args.data_dir}/oxford-iiit-pet/images/{img_name}.jpg')
                        self.train_labels.append(self.label2int[img_label])

            #testset lists
            with open(f'{self.args.data_dir}/oxford-iiit-pet/annotations/test.txt', 'r') as f_test:
                lines = f_test.readlines()
                for img in lines:
                    img_name = img.split(' ')[0]
                    img_label = img_name.rsplit('_', 1)[0]
                    if img_label in self.task_dict[self.args.tasknum]:
                        self.test_imgs.append(f'{self.args.data_dir}/oxford-iiit-pet/images/{img_name}.jpg')
                        self.test_labels.append(self.label2int[img_label])   
        
        elif self.args.data == 'svhn':
            trainset = datasets.SVHN(root=self.args.data_dir,
                            split='train', download=True,
                            transform=transforms.ToTensor())
            testset = datasets.SVHN(root=self.args.data_dir,
                            split='test', download=True,
                            transform=transforms.ToTensor())
            
            train_data = sio.loadmat(f'{self.args.data_dir}/train_32x32.mat')
            x_train = train_data['X']
            y_train = train_data['y']
            
            test_data = sio.loadmat(f'{self.args.data_dir}/test_32x32.mat')
            x_test = test_data['X']
            y_test = test_data['y']

            #for trainlist
            dest_dir = f'{self.args.data_dir}/train'
            os.makedirs(dest_dir, exist_ok=True)
            
            for idx in range(len(y_train)):
                img = x_train[:,:,:,idx]
                img_path = f'{dest_dir}/{idx}.jpg'
                img_label = y_train.flat[idx]
                if str(img_label) in self.task_dict[self.args.tasknum]:
                    cv2.imwrite(img_path, img)
                    self.train_imgs.append(img_path)
                    self.train_labels.append(img_label)
            
            #for testlist
            dest_dir = f'{self.args.data_dir}/test'
            os.makedirs(dest_dir, exist_ok=True)
            
            for idx in range(len(y_test)):
                img = x_test[:,:,:,idx]
                img_path = f'{dest_dir}/{idx}.jpg'
                img_label = y_test.flat[idx]
                if str(img_label) in self.task_dict[self.args.tasknum]:
                    cv2.imwrite(img_path, img)
                    self.test_imgs.append(img_path)
                    self.test_labels.append(img_label)
       
        elif self.args.data == 'oxfordflowers':
            pass
                
        else:
            # Download link for StanfordCars dataset is down 
            # https://github.com/pytorch/vision/issues/7545#issuecomment-1575410733
            pass

    def get_datasets(self):
        print(f"INFO : Loading {self.args.data} TRAIN & TEST data for TASK {self.args.tasknum} ... ")
        print("CLASSES : ", self.task_dict[self.args.tasknum])
        return ImageDataset(self.train_imgs, self.train_labels, 'train', self.img_processor), ImageDataset(self.test_imgs, self.test_labels, 'test', self.img_processor)


