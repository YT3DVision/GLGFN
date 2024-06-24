import argparse
import os
import random

import cv2
import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def parse_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('--dataroot', type=str, default=r'D:\DataSets\Derain_change\RealRain-1k-L')
    # parser.add_argument('--patch_size', type=int, default=192)
    # parser.add_argument('--patch_size', type=int, default=[640,640])
    return parser.parse_args()

cfg=parse_args()

"""Train SysDataset Rain100H"""
class Train_RealRain1K_L_Dataset(Dataset):
    def __init__(self):
        super(Train_RealRain1K_L_Dataset, self).__init__()
        self.root_dir = os.path.join(cfg.dataroot, 'train')

        self.rain_path = os.path.join(self.root_dir,'input')
        self.norain_path = os.path.join(self.root_dir,'target')

        self.rain_file = os.listdir(self.rain_path)
        self.norain_file = os.listdir(self.norain_path)

        self.rain_file.sort(key=lambda i: int(i[0:-4]))
        self.norain_file.sort(key=lambda i: int(i[0:-4]))

        self.patch_size = cfg.patch_size
        self.file_num = len(self.rain_file)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        rain_file_name = self.rain_file[idx]
        norain_file_name=self.norain_file[idx]

        rain_img_file = os.path.join(self.rain_path, rain_file_name)
        norain_img_file=os.path.join(self.norain_path,norain_file_name)
        rain_img=cv2.imread(rain_img_file).astype(np.float32) / 255
        norain_img=cv2.imread(norain_img_file).astype(np.float32) / 255

        rain_img = TF.to_tensor(rain_img)
        norain_img = TF.to_tensor(norain_img)

        rain_img, norain_img = self.crop(rain_img, norain_img)
        X,Y = rain_img,norain_img

        sample = {'X': X, 'Y': Y}

        return sample

    def crop(self,rain_img,norain_img):
        patch_size = self.patch_size
        c, h, w = rain_img.shape
        p_h, p_w = patch_size[0],patch_size[1]

        padw = p_w - w if w < p_w else 0
        padh = p_h - h if h < p_h else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            # rain_img = TF.pad(rain_img, (0,0,padw,padh), padding_mode='reflect')
            # norain_img = TF.pad(norain_img, (0,0,padw,padh), padding_mode='reflect')
            rain_img = TF.pad(rain_img, (0,0,padw,padh))       #左上右下
            norain_img = TF.pad(norain_img, (0,0,padw,padh))

        hh ,ww = rain_img.shape[1],rain_img.shape[2]
        r1 = random.randint(0,hh-p_h)
        r2 = random.randint(0,ww-p_w)
        X = rain_img[:,r1:r1+p_h,r2:r2+p_w]
        Y = norain_img[:,r1:r1+p_h,r2:r2+p_w]
        # rain_img = X.cpu().numpy()
        # rain_img = rain_img.transpose(1,2,0)
        # rain_img = np.uint8(rain_img * 255.0)
        # cv2.imwrite('./1.png',rain_img)
        #
        # rain_img = Y.cpu().numpy()
        # rain_img = rain_img.transpose(1,2,0)
        # rain_img = np.uint8(rain_img * 255.0)
        # cv2.imwrite('./2.png',rain_img)
        return X,Y

"""Train RealRain1K_L data Augmentation"""
class Train_RealRain1K_L_Aug_Dataset(Dataset):
    def __init__(self,patch_size):
        super(Train_RealRain1K_L_Aug_Dataset, self).__init__()
        self.root_dir = os.path.join(cfg.dataroot, 'train')

        self.rain_path = os.path.join(self.root_dir,'input')
        self.norain_path = os.path.join(self.root_dir,'target')

        self.rain_file = os.listdir(self.rain_path)
        self.norain_file = os.listdir(self.norain_path)

        self.rain_file.sort(key=lambda i: int(i[0:-4]))
        self.norain_file.sort(key=lambda i: int(i[0:-4]))

        self.patch_size = patch_size
        self.file_num = len(self.rain_file)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        rain_file_name = self.rain_file[idx]
        norain_file_name=self.norain_file[idx]

        rain_img_file = os.path.join(self.rain_path, rain_file_name)
        norain_img_file=os.path.join(self.norain_path,norain_file_name)
        rain_img=cv2.imread(rain_img_file).astype(np.float32) / 255
        norain_img=cv2.imread(norain_img_file).astype(np.float32) / 255

        rain_img = TF.to_tensor(rain_img)
        norain_img = TF.to_tensor(norain_img)

        rain_img, norain_img = self.crop(rain_img, norain_img)
        rain_img,norain_img = self.random_augmentation(rain_img,norain_img)
        X,Y = rain_img,norain_img

        sample = {'X': X, 'Y': Y}

        return sample

    def crop(self,rain_img,norain_img):
        c, h, w = rain_img.shape
        p_h, p_w = self.patch_size,self.patch_size

        padw = p_w - w if w < p_w else 0
        padh = p_h - h if h < p_h else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            # rain_img = TF.pad(rain_img, (0,0,padw,padh), padding_mode='reflect')
            # norain_img = TF.pad(norain_img, (0,0,padw,padh), padding_mode='reflect')
            rain_img = TF.pad(rain_img, (0,0,padw,padh))       #左上右下
            norain_img = TF.pad(norain_img, (0,0,padw,padh))

        hh ,ww = rain_img.shape[1],rain_img.shape[2]
        r1 = random.randint(0,hh-p_h)
        r2 = random.randint(0,ww-p_w)
        X = rain_img[:,r1:r1+p_h,r2:r2+p_w]
        Y = norain_img[:,r1:r1+p_h,r2:r2+p_w]
        # rain_img = X.cpu().numpy()
        # rain_img = rain_img.transpose(1,2,0)
        # rain_img = np.uint8(rain_img * 255.0)
        # cv2.imwrite('./1.png',rain_img)
        #
        # rain_img = Y.cpu().numpy()
        # rain_img = rain_img.transpose(1,2,0)
        # rain_img = np.uint8(rain_img * 255.0)
        # cv2.imwrite('./2.png',rain_img)
        return X,Y

    def random_augmentation(self,rain_img,norain_img):
        flag = random.randint(0,7)
        if flag == 0:
            # original
            X = rain_img
            Y = norain_img
        elif flag == 1:
            # flip up and down
            X = torch.flip(rain_img,dims=[1])
            Y = torch.flip(norain_img,dims=[1])
        elif flag == 2:
            #rotate counterwise 90 degree
            X = torch.rot90(rain_img,1,dims=[1,2])
            Y = torch.rot90(norain_img,1,dims=[1,2])
        elif flag == 3:
            #rotate counterwise 90 degree and flip up and down
            X = torch.rot90(rain_img,1,dims=[1,2])
            X = torch.flip(X,dims=[1])
            Y = torch.rot90(norain_img,1,dims=[1,2])
            Y = torch.flip(Y,dims=[1])
        elif flag == 4:
            #rotate counterwise 180 degree
            X = torch.rot90(rain_img,2,dims=[1,2])
            Y = torch.rot90(norain_img,2,dims=[1,2])
        elif flag == 5:
            #rotate counterwise 180 degree and flip up and down
            X = torch.rot90(rain_img,2,dims=[1,2])
            X = torch.flip(X,dims=[1])
            Y = torch.rot90(norain_img,2,dims=[1,2])
            Y = torch.flip(Y,dims=[1])
        elif flag == 6:
            #rotate counterwise 270 degree
            X = torch.rot90(rain_img,3,dims=[1,2])
            Y = torch.rot90(norain_img,3,dims=[1,2])
        elif flag == 7:
            #rotate counterwise 180 degree and flip up and down
            X = torch.rot90(rain_img,3,dims=[1,2])
            X = torch.flip(X,dims=[1])
            Y = torch.rot90(norain_img,3,dims=[1,2])
            Y = torch.flip(Y,dims=[1])
        else:
            raise Exception('Invalid choice of image transformation')

        # rain_img = X.cpu().numpy()
        # rain_img = rain_img.transpose(1,2,0)
        # rain_img = np.uint8(rain_img * 255.0)
        # cv2.imwrite('./1.png',rain_img)
        #
        # rain_img = Y.cpu().numpy()
        # rain_img = rain_img.transpose(1,2,0)
        # rain_img = np.uint8(rain_img * 255.0)
        # cv2.imwrite('./2.png',rain_img)
        return X,Y

"""Test RealRain1K, use whole resolution"""
class Test_RealRain1K_L_Dataset_whole_resolution(Dataset):
    def __init__(self):
        super(Test_RealRain1K_L_Dataset_whole_resolution, self).__init__()

        self.root_dir = os.path.join(cfg.dataroot, 'test')

        self.rain_path = os.path.join(self.root_dir,'input')
        self.norain_path = os.path.join(self.root_dir,'target')

        self.rain_file = os.listdir(self.rain_path)
        self.norain_file = os.listdir(self.norain_path)

        self.rain_file.sort(key = lambda i: int(i[0:-4]))
        self.norain_file.sort(key = lambda i: int(i[0:-4]))

        self.file_num = len(self.rain_file)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        rain_file_name = self.rain_file[idx]
        norain_file_name=self.norain_file[idx]

        rain_img_file = os.path.join(self.rain_path, rain_file_name)
        norain_img_file=os.path.join(self.norain_path,norain_file_name)
        rain_img=cv2.imread(rain_img_file).astype(np.float32) / 255
        norain_img=cv2.imread(norain_img_file).astype(np.float32) / 255

        rain_img = TF.to_tensor(rain_img)
        norain_img = TF.to_tensor(norain_img)

        rain_img, norain_img, padh, padw = self.handle(rain_img, norain_img)
        X,Y = rain_img,norain_img

        sample = {'X': X, 'Y': Y, 'padh': padh, 'padw': padw}

        return sample

    def handle(self, rain_img, norain_img):
        factor = 32
        c, h, w = rain_img.shape

        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0

        rain_img = TF.pad(rain_img, (0, 0, padw, padh))
        # norain_img = TF.pad(norain_img, (0, 0, padw, padh))

        X = rain_img
        Y = norain_img

        return X, Y, padh, padw

class Val_RealRain1K_L_Dataset(Dataset):
    def __init__(self):
        super(Val_RealRain1K_L_Dataset, self).__init__()

        self.root_dir = os.path.join(cfg.dataroot, 'test')

        self.rain_path = os.path.join(self.root_dir,'input')
        self.norain_path = os.path.join(self.root_dir,'target')

        self.rain_file = os.listdir(self.rain_path)
        self.norain_file = os.listdir(self.norain_path)

        self.rain_file.sort(key=lambda i: int(i[0:-4]))
        self.norain_file.sort(key=lambda i: int(i[0:-4]))

        self.patch_size = cfg.patch_size
        self.file_num = len(self.rain_file)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        rain_file_name = self.rain_file[idx]
        norain_file_name = self.norain_file[idx]

        rain_img_file = os.path.join(self.rain_path, rain_file_name)
        norain_img_file = os.path.join(self.norain_path, norain_file_name)
        rain_img = cv2.imread(rain_img_file).astype(np.float32) / 255
        norain_img = cv2.imread(norain_img_file).astype(np.float32) / 255

        rain_img = TF.to_tensor(rain_img)
        norain_img = TF.to_tensor(norain_img)

        rain_img, norain_img = self.crop(rain_img, norain_img)
        X,Y = rain_img,norain_img

        sample = {'X': X, 'Y': Y}

        return sample

    def crop(self,rain_img,norain_img):
        patch_size = 256
        c, h, w = rain_img.shape
        p_h, p_w = patch_size,patch_size

        padw = patch_size - w if w < p_w else 0
        padh = patch_size - h if h < p_h else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            # rain_img = TF.pad(rain_img, (0,0,padw,padh), padding_mode='reflect')
            # norain_img = TF.pad(norain_img, (0,0,padw,padh), padding_mode='reflect')
            rain_img = TF.pad(rain_img, (0,0,padw,padh))       #左上右下
            norain_img = TF.pad(norain_img, (0,0,padw,padh))


        hh ,ww = rain_img.shape[1],rain_img.shape[2]
        r1 = random.randint(0,hh-p_h)
        r2 = random.randint(0,ww-p_w)
        X = rain_img[:,r1:r1+p_h,r2:r2+p_w]
        Y = norain_img[:,r1:r1+p_h,r2:r2+p_w]

        return X,Y


if __name__ == '__main__':

    traindataset = Train_RealRain1K_L_Aug_Dataset()
    # print(len(traindataset))
    for i in range(len(traindataset)):
        smp = traindataset[i]
        # break
        print(i)
        for k,v in smp.items():
            print(k,v.shape)
        print('\n')

