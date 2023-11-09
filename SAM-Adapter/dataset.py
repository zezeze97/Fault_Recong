""" train and test dataset

author jundewu
"""
import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from utils import random_click
import random
from monai.transforms import LoadImaged, Randomizable,LoadImage


class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }

class Fault2D(Dataset):
    def __init__(self, args, data_path ,mode = 'train',prompt = 'click', plane = False):

        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transforms.Compose([transforms.RandomCrop((args.image_size, args.image_size), pad_if_needed=True, fill=0),
                                             transforms.ToTensor(),
                                             # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                             ])
        self.transform_msk = transforms.Compose([transforms.RandomCrop((args.image_size, args.image_size), pad_if_needed=True, fill=0),
                                                 transforms.Resize((args.out_size,args.out_size), interpolation=Image.NEAREST),
                                                 transforms.ToTensor(),
                                                ])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        inout = 1
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        msk_path = os.path.join(self.data_path, self.mode, 'ann', name.replace('.npy', '.png'))
        
        # load seis and force 3 channel
        seis = np.load(img_path)
        # cvt to 0-255
        seis =  255 * (seis - seis.min()) / (seis.max() - seis.min())
        seis = np.stack([seis, seis, seis], axis=2)
        
        # cvt to Image obj
        img = Image.fromarray(seis.astype(np.uint8))
        mask = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        mask = Image.fromarray(mask * 255)

        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        
        state = torch.get_rng_state()
        img = self.transform(img)
        torch.set_rng_state(state)
        mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }
