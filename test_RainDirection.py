
import argparse

import numpy as np
import math
import cv2

from model.Baseline import Baseline
from utils.cal_ssim import SSIM
from data.Data_RainDirection import Test_RainDirection
import torch


from torch.utils.data import DataLoader
import os


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./training_model/RainDirection')
    parser.add_argument('--device',   type=str,default='cuda:0')

    parser.add_argument('--ckp_name' ,type=str ,default='best')
    parser.add_argument('--result_dir', type=str,default='./results/RainDirection')


    return parser.parse_args()

cfg = parse_args()

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

# def save_image(derain,idx):
def save_image(type,img, idx):
    #创建保存result的文件夹
    ensure_dir(cfg.result_dir)
    derain_save = img
    derain_save = torch.clamp(derain_save, 0.0, 1.0)
    derain_save_np = derain_save.cpu().numpy()
    derain_save_np = derain_save_np.squeeze()
    derain_save_np = derain_save_np.transpose(1, 2, 0)
    derain_save_np = np.uint8(derain_save_np * 255.0)
    num=str(idx+1).zfill(3)
    print(os.path.join(os.path.join(cfg.result_dir, '%s_%s.png'%(type,num))))
    cv2.imwrite(os.path.join(os.path.join(cfg.result_dir, '%s_%s.png'%(type,num))), derain_save_np)

def PSNR(img1, img2):
    b, _, _, _ = img1.shape
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)  # +mse
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class Session:
    def __init__(self):
        self.device = cfg.device
        self.model_dir=cfg.model_dir
        self.net = Baseline().to(self.device)
        self.ssim=SSIM().to(self.device)


    def get_dataloader(self):
        dataset = Test_RainDirection()
        dataloader = DataLoader(dataset,batch_size = 1, shuffle=False, num_workers=1, drop_last=False)
        return dataloader

    def load_checkpoints(self,name):

        ckp_path = os.path.join(self.model_dir, name)
        try:
            print('Loading checkpoint %s' % ckp_path)
            ckp = torch.load(ckp_path)
        except FileNotFoundError:
            print('No checkpoint %s' % ckp_path)
            return
        self.net.load_state_dict(ckp['net'])

        self.start_epoch = ckp['initial_epoch']
        # print('Loading %d epoch....'%self.start_epoch)

    def inf_batch(self,batch):
        X,Y = batch['X'].to(self.device),batch['Y'].to(self.device)
        padh = int(batch['padh'])
        padw = int(batch['padw'])

        with torch.no_grad():
            # _,derain,refine = self.net(X)
            _,_,derain = self.net(X)
            b, c, h, w = derain.shape
            derain = derain[: , : , 0:h - padh , 0:w - padw]
            # refine = refine[: , : , 0:h - padh , 0:w - padw]

        derain_ssim = self.ssim(derain,Y)
        derain_psnr = PSNR(derain.data.cpu().numpy()*255,Y.data.cpu().numpy()*255)
        #
        # refine_ssim = self.ssim(refine,Y)
        # refine_psnr = PSNR(refine.data.cpu().numpy()*255,Y.data.cpu().numpy()*255)
        # return derain_ssim,derain_psnr,refine_ssim,refine_psnr,derain,refine
        return derain_ssim,derain_psnr,derain


def run_test():
    sess=Session()
    sess.net.eval()
    sess.load_checkpoints(cfg.ckp_name)
    dataloader=sess.get_dataloader()
    derain_psnr_all = 0
    derain_ssim_all = 0
    # refine_psnr_all = 0
    # refine_ssim_all = 0
    all_num = 0

    for i, batch in enumerate(dataloader):
        # derain_ssim,derain_psnr,refine_ssim,refine_psnr,derain,refine = sess.inf_batch(batch)
        derain_ssim,derain_psnr,derain = sess.inf_batch(batch)
        print('num:%d   derain_psnr:%5f  derain_ssim:%5f'%(i,derain_psnr,derain_ssim))
        # print('num:%d   refine_psnr:%5f  refine_ssim:%5f'%(i,refine_psnr,refine_ssim))

        logfile = open('./results/' + 'test_RainDirection' + '.txt', 'a+')
        logfile.write(
            'num   = ' + str(i + 1)    + '\t'
            'derain_psnr  = ' + str(derain_psnr) + '\t'
            'derain_ssim  = ' + str(derain_ssim) + '\t'
            # '\n'
            # 'refine_psnr  = ' + str(refine_psnr) + '\t'
            # 'refine_ssim  = ' + str(refine_ssim) + '\t'
            '\n\n'
        )

        save_image('derain',derain,i)
        # save_image('refine',refine,i)

        derain_psnr_all = derain_psnr_all + derain_psnr
        derain_ssim_all = derain_ssim_all + derain_ssim

        # refine_psnr_all = refine_psnr_all + refine_psnr
        # refine_ssim_all = refine_ssim_all + refine_ssim
        all_num += 1
    print(all_num)

    print('derain_psnr_ll:%8f'%(derain_psnr_all/all_num))
    print('derain_ssim_ll:%8f'%(derain_ssim_all/all_num))

    # print('refine_psnr_ll:%8f'%(refine_psnr_all/all_num))
    # print('refine_ssim_ll:%8f'%(refine_ssim_all/all_num))

    logfile = open('./results/' + 'test_RainDirection' + '.txt', 'a+')

    logfile.write(
    'all_num      = ' + str(all_num)     + '\t'
    '\n'
    'derain_psnr_ll  = ' + str(derain_psnr_all/all_num) + '\t'
    'derain_ssim_ll  = ' + str(derain_ssim_all/all_num) + '\t'
    # '\n'
    # 'refine_psnr_ll  = ' + str(refine_psnr_all/all_num) + '\t'
    # 'refine_ssim_ll  = ' + str(refine_ssim_all/all_num) + '\t'
    '\n\n'
)

if __name__ == '__main__':
    run_test()
