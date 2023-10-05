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
                            },
            
            'stanfordcars' : {   0 : [ 'Dodge Ram Pickup 3500 Crew Cab 2010','Cadillac CTS-V Sedan 2012', 'Audi S5 Convertible 2012',
                                    'Ram C-V Cargo Van Minivan 2012', 'smart fortwo Convertible 2012', 'Audi V8 Sedan 1994', 'Suzuki Kizashi Sedan 2012',
                                    'Chevrolet Express Van 2007', 'Chrysler Town and Country Minivan 2012', 'Rolls-Royce Phantom Sedan 2012','Suzuki SX4 Hatchback 2012', 
                                    'Lamborghini Aventador Coupe 2012', 'Chevrolet Silverado 1500 Classic Extended Cab 2007', 'Ford F-450 Super Duty Crew Cab 2012', 'Hyundai Veloster Hatchback 2012',
                                    'Dodge Magnum Wagon 2008', 'GMC Canyon Extended Cab 2012', 'Infiniti QX56 SUV 2011', 'Lamborghini Gallardo LP 570-4 Superleggera 2012'],
                                                
                                1 : [ 'Buick Regal GS 2012', 'Aston Martin V8 Vantage Convertible 2012', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
                                        'Audi TT Hatchback 2011', 'Chevrolet Malibu Hybrid Sedan 2010','Honda Odyssey Minivan 2012', 'BMW X6 SUV 2012',
                                        'Audi 100 Wagon 1994', 'Volvo 240 Sedan 1993', 'Plymouth Neon Coupe 1999','Chevrolet Sonic Sedan 2012',
                                        'Chrysler Sebring Convertible 2010', 'Chevrolet Silverado 2500HD Regular Cab 2012', 'Chevrolet Cobalt SS 2010', 'HUMMER H2 SUT Crew Cab 2009',
                                        'Scion xD Hatchback 2012', 'Spyker C8 Coupe 2009', 'Ford E-Series Wagon Van 2012', 'Toyota 4Runner SUV 2012'],
                                
                                2 : [ 'Ford Focus Sedan 2007','Ford GT Coupe 2006', 'McLaren MP4-12C Coupe 2012','Dodge Caliber Wagon 2012',
                                        'Acura TSX Sedan 2012', 'Chrysler 300 SRT-8 2010', 'Jeep Wrangler SUV 2012','Chevrolet Monte Carlo Coupe 2007',
                                        'Chevrolet HHR SS 2010','Dodge Dakota Crew Cab 2010',  'Bentley Continental GT Coupe 2012', 
                                        'Porsche Panamera Sedan 2012', 'Nissan NV Passenger Van 2012', 'Audi S6 Sedan 2011', 'Nissan 240SX Coupe 1998',
                                        'Toyota Camry Sedan 2012', 'Acura RL Sedan 2012', 'Spyker C8 Convertible 2009', 'BMW M5 Sedan 2010'],
                                
                                3 : [ 'Jeep Compass SUV 2012', 'Hyundai Veracruz SUV 2012', 'Buick Verano Sedan 2012', 'Isuzu Ascender SUV 2008',
                                        'Hyundai Accent Sedan 2012', 'Audi TT RS Coupe 2012', 'Geo Metro Convertible 1993', 'Buick Enclave SUV 2012',
                                        'Mercedes-Benz 300-Class Convertible 1993', 'Dodge Caliber Wagon 2007',  'Ford Mustang Convertible 2007',
                                        'Hyundai Sonata Hybrid Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012', 'Hyundai Santa Fe SUV 2012', 'BMW 1 Series Convertible 2012',
                                        'Ford Fiesta Sedan 2012', 'Dodge Charger SRT-8 2009', 'Aston Martin Virage Convertible 2012', 'Ford Freestar Minivan 2007'],
                                
                                4 : [ 'Hyundai Genesis Sedan 2012', 'Jeep Liberty SUV 2012','Chevrolet Malibu Sedan 2007', 'Hyundai Sonata Sedan 2012',
                                        'Chevrolet Traverse SUV 2012','BMW M6 Convertible 2010','Bentley Mulsanne Sedan 2011', 'Chevrolet Impala Sedan 2007',
                                        'Jaguar XK XKR 2012','Dodge Challenger SRT8 2011','Ford F-150 Regular Cab 2012', 'GMC Acadia SUV 2012', 'Lamborghini Diablo Coupe 2001',  
                                        'Land Rover LR2 SUV 2012', 'Dodge Dakota Club Cab 2007', 'AM General Hummer SUV 2000', 'Aston Martin V8 Vantage Coupe 2012', 
                                        'Volkswagen Golf Hatchback 2012', 'Ferrari 458 Italia Convertible 2012', 'Audi A5 Coupe 2012'],
                                
                                5 : [ 'Infiniti G Coupe IPL 2012','Bugatti Veyron 16.4 Coupe 2009','Ferrari 458 Italia Coupe 2012',
                                        'Acura ZDX Hatchback 2012','Hyundai Elantra Touring Hatchback 2012','Suzuki Aerio Sedan 2007','Ford F-150 Regular Cab 2007',
                                        'BMW Z4 Convertible 2012','Chevrolet Corvette ZR1 2012','Rolls-Royce Ghost Sedan 2012','Honda Accord Sedan 2012',
                                        'Volvo C30 Hatchback 2012', 'Dodge Journey SUV 2012', 'HUMMER H3T Crew Cab 2010', 'Chevrolet Silverado 1500 Extended Cab 2012',
                                        'Dodge Ram Pickup 3500 Quad Cab 2009', 'Volkswagen Golf Hatchback 1991', 'Dodge Durango SUV 2007', 'Ford Edge SUV 2012', 'Ford Expedition EL SUV 2009'],
                                
                                6 : [ 'Audi S5 Coupe 2012','Audi S4 Sedan 2012','Lincoln Town Car Sedan 2011','Jeep Grand Cherokee SUV 2012',
                                        'Lamborghini Reventon Coupe 2008','Chevrolet Express Cargo Van 2007','Mitsubishi Lancer Sedan 2012', 'Chevrolet Corvette Ron Fellows Edition Z06 2007', 
                                        'Toyota Corolla Sedan 2012','GMC Terrain SUV 2012','Toyota Sequoia SUV 2012','Nissan Juke Hatchback 2012',
                                        'Ferrari FF Coupe 2012', 'Honda Odyssey Minivan 2007', 'Hyundai Elantra Sedan 2007', 'Ford Ranger SuperCab 2011',
                                        'Nissan Leaf Hatchback 2012','Dodge Charger Sedan 2012', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'Volkswagen Beetle Hatchback 2012' ],
                                
                                7 : [ 'Chevrolet Corvette Convertible 2012','GMC Yukon Hybrid SUV 2012','Land Rover Range Rover SUV 2012', 'Cadillac Escalade EXT Crew Cab 2007', 
                                        'BMW 3 Series Sedan 2012','Chevrolet Tahoe Hybrid SUV 2012','Acura TL Type-S 2008','BMW ActiveHybrid 5 Sedan 2012',
                                        'Bugatti Veyron 16.4 Convertible 2009','Dodge Durango SUV 2012','Aston Martin Virage Coupe 2012', 'Acura Integra Type R 2001',                       
                                        'Dodge Sprinter Cargo Van 2009',  'Honda Accord Coupe 2012', 'Ferrari California Convertible 2012',
                                        'BMW 6 Series Convertible 2007', 'Chevrolet Silverado 1500 Regular Cab 2012','Audi S4 Sedan 2007', 'Jeep Patriot SUV 2012', 'Chevrolet Avalanche Crew Cab 2012'],
                                
                                8 : [ 'Suzuki SX4 Sedan 2012','Chrysler Crossfire Convertible 2008','Audi 100 Sedan 1994','Audi TTS Coupe 2012',
                                        'Chevrolet TrailBlazer SS 2009', 'Audi R8 Coupe 2012', 'Eagle Talon Hatchback 1998', 'Bentley Continental Supersports Conv. Convertible 2012',
                                        'Mercedes-Benz SL-Class Coupe 2009', 'Volvo XC90 SUV 2007', 'Mercedes-Benz C-Class Sedan 2012', 'BMW 1 Series Coupe 2012',
                                        'Bentley Continental GT Coupe 2007','Buick Rainier SUV 2007','Mazda Tribute SUV 2011','BMW X5 SUV 2007',
                                        'BMW M3 Coupe 2012','FIAT 500 Abarth 2012', 'Chevrolet Camaro Convertible 2012', 'MINI Cooper Roadster Convertible 2012' ],
                                
                                9 : [ 'GMC Savana Van 2012', 'Chrysler PT Cruiser Convertible 2008', 'Fisker Karma Sedan 2012', 'Tesla Model S Sedan 2012',
                                        'BMW 3 Series Wagon 2012', 'Mercedes-Benz Sprinter Van 2012', 'Hyundai Azera Sedan 2012', 'Chrysler Aspen SUV 2009',
                                        'Acura TL Sedan 2012', 'Audi RS 4 Convertible 2008', 'Daewoo Nubira Wagon 2002', 'Mercedes-Benz E-Class Sedan 2012',
                                        'Dodge Caravan Minivan 1997', 'Bentley Continental Flying Spur Sedan 2007', 'Cadillac SRX SUV 2012', 'Maybach Landaulet Convertible 2012',
                                        'FIAT 500 Convertible 2012', 'Bentley Arnage Sedan 2009', 'BMW X3 SUV 2012','Hyundai Tucson SUV 2012']
                            
            },

            'cifar10' : {   0 : [0, 1], 
                            1 : [2, 3],
                            2 : [4, 5], 
                            3 : [6, 7], 
                            4 : [8, 9] }


  

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
        if isinstance(self.image_list[idx], str):
            img = self.transform(Image.open(self.image_list[idx]).convert('RGB'))
        else:
            img = self.transform(self.image_list[idx])
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

        elif self.args.data == 'stanfordcars':
            # https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder

            # for trainlist
            for class_name in self.task_dict[self.args.tasknum]:
                class_dir = f'{self.args.data_dir}/car_data/car_data/train/{class_name}'
                img_paths = glob.glob(f'{class_dir}/*.jpg')
                img_labels = [self.label2int[class_name] for i in range(len(img_paths))]
                self.train_imgs += img_paths
                self.train_labels += img_labels
            assert len(self.train_imgs) == len(self.train_labels)

            # for testlist
            for class_name in self.task_dict[self.args.tasknum]:
                class_dir = f'{self.args.data_dir}/car_data/car_data/test/{class_name}'
                img_paths = glob.glob(f'{class_dir}/*.jpg')
                img_labels = [self.label2int[class_name] for i in range(len(img_paths))]
                self.test_imgs += img_paths
                self.test_labels += img_labels
            assert len(self.test_imgs) == len(self.test_labels)
        
        elif self.args.data == 'cifar10':
            trainset = datasets.CIFAR10(root=self.args.data_dir, train=True, download=True)
            testset = datasets.CIFAR10(root=self.args.data_dir, train=False, download=True)

            # for trainlist
            for img, label in trainset:
                if label in self.task_dict[self.args.tasknum]:
                    self.train_imgs.append(img)
                    self.train_labels.append(label)

    
            # for trainlist
            for img, label in testset:
                if label in self.task_dict[self.args.tasknum]:
                    self.test_imgs.append(img)
                    self.test_labels.append(label)

    def get_datasets(self):
        print(f"INFO : Loading {self.args.data} TRAIN : {len(self.train_labels)} & TEST : {len(self.test_labels)} data for TASK {self.args.tasknum} ... ")
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
                img_label = y_train.flat[idx] - 1

                if label_freqs[img_label]+1 <= self.n_samples:
                    img = x_train[:,:,:,idx]
                    img_path = f'{dest_dir}/{idx}.jpg'
                    cv2.imwrite(img_path, img)
                    self.train_imgs.append(img_path)
                    self.train_labels.append(img_label)
                    label_freqs[img_label] += 1
                    
                if sum(label_freqs) == self.num_classes * self.n_samples:
                    print("Total training samples : ", len(self.train_labels))
                    break
            
            #for testlist
            dest_dir = f'{self.data_dir}/test'
            os.makedirs(dest_dir, exist_ok=True)
            
            for idx in range(len(y_test)):
                img = x_test[:,:,:,idx]
                img_path = f'{dest_dir}/{idx}.jpg'
                img_label = y_test.flat[idx] - 1
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

        elif self.data == 'stanfordcars':
            # https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder

            class_names_list = os.listdir(f'{self.data_dir}/car_data/car_data/train')

            # for trainlist
            for class_name in class_names_list:
                class_dir = f'{self.data_dir}/car_data/car_data/train/{class_name}'
                img_paths = random.sample(glob.glob(f'{class_dir}/*.jpg'), self.n_samples)
                img_labels = [self.label2int[class_name] for i in range(len(img_paths))]
                self.train_imgs += img_paths
                self.train_labels += img_labels
            assert len(self.train_imgs) == len(self.train_labels)
            print("Total training samples : ", len(self.train_labels))

            # for testlist

            for class_name in class_names_list:
                class_dir = f'{self.data_dir}/car_data/car_data/test/{class_name}'
                img_paths = glob.glob(f'{class_dir}/*.jpg')
                img_labels = [self.label2int[class_name] for i in range(len(img_paths))]
                self.test_imgs += img_paths
                self.test_labels += img_labels
            assert len(self.test_imgs) == len(self.test_labels)

        elif self.data == 'cifar10':
            trainset = datasets.CIFAR10(root=self.data_dir, train=True, download=True)
            testset = datasets.CIFAR10(root=self.data_dir, train=False, download=True)

            # for trainlist
            while True:
                img, img_label = random.choice(trainset)
                # img_label = y_train.flat[idx] - 1

                if label_freqs[img_label]+1 <= self.n_samples:
                    self.train_imgs.append(img)
                    self.train_labels.append(img_label)
                    label_freqs[img_label] += 1
                    
                if sum(label_freqs) == self.num_classes * self.n_samples:
                    print("Total training samples : ", len(self.train_labels))
                    break
    
            # for trainlist
            for img, label in testset:
                # if label in self.task_dict[self.args.tasknum]:
                self.test_imgs.append(img)
                self.test_labels.append(label)
    

    def get_datasets(self):
        print(f"INFO : Loading {self.data} TRAIN : {len(self.train_labels)} & TEST : {len(self.test_labels)} data ... ")
        return ImageDataset(self.train_imgs, self.train_labels, 'train', self.img_processor), ImageDataset(self.test_imgs, self.test_labels, 'test', self.img_processor)
