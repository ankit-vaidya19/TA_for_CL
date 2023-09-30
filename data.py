import torch
import glob
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import cv2
import random

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
            
            'svhn' : {  0 : ['0', '1'],
                        1 : ['2', '3'],
                        2 : ['4', '5'],
                        3 : ['6', '7'],
                        4 : ['8', '9']
                        },
            
            'oxfordflowers' : { 0 : [x for x in range(1,11)] , 1 : [x for x in range(11, 21)],
                                2 : [x for x in range(21,31)], 3 : [x for x in range(31, 41)],
                                4 : [x for x in range(41,51)], 5 : [x for x in range(51, 61)],
                                6 : [x for x in range(61,71)], 7 : [x for x in range(71, 81)],
                                8 : [x for x in range(81,91)], 9 : [x for x in range(91, 103)]
                            }

            }

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
                img_label = y_train.flat[idx]-1
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
                img_label = y_test.flat[idx]-1
                if str(img_label) in self.task_dict[self.args.tasknum]:
                    cv2.imwrite(img_path, img)
                    self.test_imgs.append(img_path)
                    self.test_labels.append(img_label)
       
        elif self.args.data == 'oxfordflowers':
            trainset = datasets.Flowers102(root=self.args.data_dir,
                            split='train', download=True,
                            transform=transforms.ToTensor())
            testset = datasets.Flowers102(root=self.args.data_dir,
                            split='test', download=True,
                            transform=transforms.ToTensor())

            labels = sio.loadmat(f'{self.args.data_dir}/flowers-102/imagelabels.mat')
            labels = labels['labels'][0]
            data_splits = sio.loadmat(f'{self.args.data_dir}/flowers-102/setid.mat')
            x_train = data_splits['trnid'][0]
            x_val = data_splits['valid'][0]
            x_test = data_splits['tstid'][0]
            
            # trainset lists
            for img_id in x_train:
                img_path = f'{self.args.data_dir}/flowers-102/jpg/image_{str(img_id).zfill(5)}.jpg'
                img_label = labels[img_id-1]
                if img_label in self.task_dict[self.args.tasknum]:
                    self.train_imgs.append(img_path)
                    self.train_labels.append(self.label2int[img_label])
            for img_id in x_val:
                img_path = f'{self.args.data_dir}/flowers-102/jpg/image_{str(img_id).zfill(5)}.jpg'
                img_label = labels[img_id-1]
                if img_label in self.task_dict[self.args.tasknum]:
                    self.train_imgs.append(img_path)
                    self.train_labels.append(self.label2int[img_label])
            
            # testset lists
            for img_id in x_test:
                img_path = f'{self.args.data_dir}/flowers-102/jpg/image_{str(img_id).zfill(5)}.jpg'
                img_label = labels[img_id-1]
                if img_label in self.task_dict[self.args.tasknum]:
                    self.test_imgs.append(img_path)
                    self.test_labels.append(self.label2int[img_label])

        else:
            # Download link for StanfordCars dataset is down 
            # https://github.com/pytorch/vision/issues/7545#issuecomment-1575410733
            pass

    def get_datasets(self):
        print(f"INFO : Loading {self.args.data} TRAIN & TEST data for TASK {self.args.tasknum} ... ")
        print("CLASSES : ", self.task_dict[self.args.tasknum])
        return ImageDataset(self.train_imgs, self.train_labels, 'train', self.img_processor), ImageDataset(self.test_imgs, self.test_labels, 'test', self.img_processor)


class FewShotDataset():
    def __init__(self, dataset_name, data_directory, img_processor, samples_per_class=10, task_dict=task_dict):
        self.n_samples = samples_per_class
        self.data = dataset_name
        self.data_dir = data_directory
        self.img_processor = img_processor
        self.task_dict = task_dict[self.data]
        self.label2int = self.get_label2int()
        self.num_classes = len(self.label2int)
        self.train_imgs, self.train_labels = [],[]
        self.test_imgs, self.test_labels = [],[]
        self.get_lists()
        print(f"\nINFO: Taken {self.n_samples} samples for every class.")

    def get_label2int(self):
        label2int = []
        for task in self.task_dict.values():
            for clas in task:
                label2int.append(clas)

        keys = [x for x in range(len(label2int))]
        label2int = dict(zip(label2int, keys))
        return label2int

    def get_lists(self):

        label_freqs = [0 for i in range(self.num_classes)]
        
        if self.data == 'oxfordpet':
            oxfordpet = datasets.OxfordIIITPet(root=self.data_dir, download=True)

            #trainset lists
            with open(f'{self.data_dir}/oxford-iiit-pet/annotations/trainval.txt', 'r') as f_train:
                lines = f_train.readlines()
                # print(lines)

                while True:
                    img = random.choice(lines)
                    img_name = img.split(' ')[0]
                    img_label = img_name.rsplit('_', 1)[0]

                    if label_freqs[self.label2int[img_label]]+1 <= self.n_samples:
                        self.train_imgs.append(f'{self.data_dir}/oxford-iiit-pet/images/{img_name}.jpg')
                        self.train_labels.append(self.label2int[img_label])
                        label_freqs[self.label2int[img_label]] += 1
                    
                    if sum(label_freqs) == self.num_classes * self.n_samples:
                        print("Total training samples : ", len(self.train_labels))
                        break

            #testset lists
            with open(f'{self.data_dir}/oxford-iiit-pet/annotations/test.txt', 'r') as f_test:
                lines = f_test.readlines()
                for img in lines:
                    img_name = img.split(' ')[0]
                    img_label = img_name.rsplit('_', 1)[0]
                    self.test_imgs.append(f'{self.data_dir}/oxford-iiit-pet/images/{img_name}.jpg')
                    self.test_labels.append(self.label2int[img_label])   
        
        elif self.data == 'svhn':
            trainset = datasets.SVHN(root=self.data_dir,
                            split='train', download=True,
                            transform=transforms.ToTensor())
            testset = datasets.SVHN(root=self.data_dir,
                            split='test', download=True,
                            transform=transforms.ToTensor())
            
            train_data = sio.loadmat(f'{self.data_dir}/train_32x32.mat')
            x_train = train_data['X']
            y_train = train_data['y']
            
            test_data = sio.loadmat(f'{self.data_dir}/test_32x32.mat')
            x_test = test_data['X']
            y_test = test_data['y']

            #for trainlist
            dest_dir = f'{self.data_dir}/train'
            os.makedirs(dest_dir, exist_ok=True)

            while True:
                idx = random.choice([i for i in range(len(y_train))])
                img_label = str(y_train.flat[idx] - 1)

                if label_freqs[self.label2int[img_label]]+1 <= self.n_samples:
                    img = x_train[:,:,:,idx]
                    img_path = f'{dest_dir}/{idx}.jpg'
                    cv2.imwrite(img_path, img)
                    self.train_imgs.append(img_path)
                    self.train_labels.append(img_label)
                    label_freqs[self.label2int[img_label]] += 1
                    
                if sum(label_freqs) == self.num_classes * self.n_samples:
                    print("Total training samples : ", len(self.train_labels))
                    break
            
            #for testlist
            dest_dir = f'{self.data_dir}/test'
            os.makedirs(dest_dir, exist_ok=True)
            
            for idx in range(len(y_test)):
                img = x_test[:,:,:,idx]
                img_path = f'{dest_dir}/{idx}.jpg'
                img_label = y_test.flat[idx]
                cv2.imwrite(img_path, img)
                self.test_imgs.append(img_path)
                self.test_labels.append(img_label)
       
        elif self.data == 'oxfordflowers':
            trainset = datasets.Flowers102(root=self.data_dir,
                            split='train', download=True,
                            transform=transforms.ToTensor())
            testset = datasets.Flowers102(root=self.data_dir,
                            split='test', download=True,
                            transform=transforms.ToTensor())

            labels = sio.loadmat(f'{self.data_dir}/flowers-102/imagelabels.mat')
            labels = labels['labels'][0]
            data_splits = sio.loadmat(f'{self.data_dir}/flowers-102/setid.mat')
            x_train = data_splits['trnid'][0]
            x_val = data_splits['valid'][0]
            x_test = data_splits['tstid'][0]
            
            # trainset lists

            while True:
                img_id = random.choice(x_train)
                img_path = f'{self.data_dir}/flowers-102/jpg/image_{str(img_id).zfill(5)}.jpg'
                img_label = labels[img_id-1]

                if label_freqs[self.label2int[img_label]]+1 <= self.n_samples:
                    self.train_imgs.append(img_path)
                    self.train_labels.append(self.label2int[img_label])
                    label_freqs[self.label2int[img_label]] += 1

                if sum(label_freqs) == self.num_classes * self.n_samples:
                    print("Total training samples : ", len(self.train_labels))
                    break

            
            # testset lists
            for img_id in x_test:
                img_path = f'{self.data_dir}/flowers-102/jpg/image_{str(img_id).zfill(5)}.jpg'
                img_label = labels[img_id-1]
                self.test_imgs.append(img_path)
                self.test_labels.append(self.label2int[img_label])

        else:
            # Download link for StanfordCars dataset is down 
            # https://github.com/pytorch/vision/issues/7545#issuecomment-1575410733
            pass
    

    def get_datasets(self):
        print(f"INFO : Loading {self.data} TRAIN & TEST data ... ")
        return ImageDataset(self.train_imgs, self.train_labels, 'train', self.img_processor), ImageDataset(self.test_imgs, self.test_labels, 'test', self.img_processor)
