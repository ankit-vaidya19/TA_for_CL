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
    },
  
  "stanfordcars" : { 
        0: (0, 19), 1: (19, 38),
        2: (38, 57), 3: (57, 76),
        4: (76, 96), 5: (96, 116),
        6: (116, 136), 7: (136, 156),
        8: (156, 176), 9: (176, 196),                        
    }

}


def get_model(args, pretrained_checkpoint, list_of_task_checkpoints, scaling_coef, return_trainable=False):
    task_vector_list = [
        TaskVector(pretrained_checkpoint, task_checkpoint)
        for task_checkpoint in list_of_task_checkpoints
    ]
    vector_sum = sum(task_vector_list)
    model = vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    with torch.no_grad():
        for task_idx, ckpt in enumerate(sorted(list_of_task_checkpoints)):
            finetuned_weights = torch.load(ckpt)
            # task_model = ViT_LoRA(args, use_)
            
            taskwise_model = ViT_LoRA(args, use_LoRA=True)
            taskwise_model.load_state_dict(finetuned_weights)
            taskwise_model.eval()
            
            for param in taskwise_model.parameters():
                param.requires_grad = False
            # print(taskwise_model.linear.weight.shape)
            start_idx = ranges_of_classes[args.data][task_idx][0]
            end_idx = ranges_of_classes[args.data][task_idx][1]
            model.linear.weight[start_idx:end_idx , :] = taskwise_model.linear.weight[start_idx:end_idx , :]
            model.linear.bias[start_idx:end_idx] = taskwise_model.linear.bias[start_idx:end_idx]
            # print(model.linear.weight.shape)
            # print(model.linear.bias.shape)
    if return_trainable:
        for param in model.parameters():
            param.requires_grad = True
    return model
