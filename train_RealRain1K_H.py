import os
import argparse
import math
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.Data_RealRain1K_H import Train_RealRain1K_H_Aug_Dataset, Test_RealRain1K_H_Dataset_whole_resolution
import numpy as np
from model.Baseline import Baseline
from utils.cal_ssim import SSIM
import torch.nn.functional as F

"""为CPU\GPU设置种子，保证每次的随机初始化都是相同的，从而保证结果可以复现。"""
torch.cuda.manual_seed(66)
torch.manual_seed(66)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',         type=str,       default='cuda:0')
    parser.add_argument('--lr',             type=float,     default=3e-4, )
    parser.add_argument('--lr_min',         type=float,     default=2e-6, )
    parser.add_argument('--n_epochs',       type=int,       default=1000,        help='number of epochs to train')

    # parser.add_argument('--train_patch_size',type=int,      default = [128,256,384,512,640])
    # parser.add_argument('--train_batch_size',type=int,      default = [24,12,5,3,2])
    # parser.add_argument('--train_milestone',type=int,      default = [250,500,700,800])
    parser.add_argument('--train_patch_size',type=int,      default = [128,256,384,512,640,768])
    parser.add_argument('--train_batch_size',type=int,      default = [24,12,5,3,2,1])
    parser.add_argument('--train_milestone',type=int,      default = [250,500,700,800,900])

    parser.add_argument('--val_batch_size', type=int,      default=1)
    parser.add_argument('--num_workers',    type=int,       default=1)
    parser.add_argument('--val_freq',       type=int,       default=10)


    parser.add_argument('--ckp_name' ,      type=str ,      default='best')
    parser.add_argument('--model_dir',      type=str,       default='./training_model/RealRain1K_H_Baseline')
    parser.add_argument('--runs_dir',      type=str,       default='./runs/RealRain1K_H_Baseline')

    return parser.parse_args()
cfg = parse_args()


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def PSNR(img1, img2):
    b, _, _, _ = img1.shape
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class Session():
    def __init__(self):
        ensure_dir(cfg.model_dir)
        self.tb_writer = SummaryWriter(log_dir=cfg.runs_dir)
        self.device = cfg.device
        self.net = Baseline().to(self.device)
        self.l1_loss = nn.L1Loss().to(self.device)
        self.ssim_loss = SSIM().to(self.device)

        self.initial_epoch = 1
        self.n_epochs = cfg.n_epochs
        self.train_patch_size = cfg.train_patch_size
        self.train_batch_size = cfg.train_batch_size
        self.train_milestone = cfg.train_milestone

        self.optimizer = Adam(self.net.parameters(), lr=cfg.lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.n_epochs + 1, T_mult=1, eta_min=cfg.lr_min)

        self.model_dir = cfg.model_dir

    def get_train_dataloader(self,epoch):
        if(epoch > 0 and epoch <= self.train_milestone[0]):
            dataset = Train_RealRain1K_H_Aug_Dataset(self.train_patch_size[0])
            dataloader = DataLoader(dataset, batch_size = self.train_batch_size[0], shuffle=True, num_workers=cfg.num_workers, drop_last=True)
        elif(epoch > self.train_milestone[0] and epoch <= self.train_milestone[1]):
            dataset = Train_RealRain1K_H_Aug_Dataset(self.train_patch_size[1])
            dataloader = DataLoader(dataset, batch_size = self.train_batch_size[1], shuffle=True, num_workers=cfg.num_workers, drop_last=True)
        elif (epoch > self.train_milestone[1] and epoch <= self.train_milestone[2]):
            dataset = Train_RealRain1K_H_Aug_Dataset(self.train_patch_size[2])
            dataloader = DataLoader(dataset, batch_size=self.train_batch_size[2], shuffle=True,num_workers=cfg.num_workers, drop_last=True)
        elif (epoch > self.train_milestone[2] and epoch <= self.train_milestone[3]):
            dataset = Train_RealRain1K_H_Aug_Dataset(self.train_patch_size[3])
            dataloader = DataLoader(dataset, batch_size=self.train_batch_size[3], shuffle=True,num_workers=cfg.num_workers, drop_last=True)
        elif (epoch > self.train_milestone[3] and epoch <= self.train_milestone[4]):
            dataset = Train_RealRain1K_H_Aug_Dataset(self.train_patch_size[4])
            dataloader = DataLoader(dataset, batch_size=self.train_batch_size[4], shuffle=True,num_workers=cfg.num_workers, drop_last=True)
        else:
            dataset = Train_RealRain1K_H_Aug_Dataset(self.train_patch_size[5])
            dataloader = DataLoader(dataset, batch_size=self.train_batch_size[5], shuffle=True,num_workers=cfg.num_workers, drop_last=True)
        return dataloader

    def get_val_dataloader(self,):
        dataset = Test_RealRain1K_H_Dataset_whole_resolution()
        dataloader = DataLoader(dataset,batch_size=cfg.val_batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
        return dataloader

    def load_checkpoint_net(self,name):
        ckp_path = os.path.join(self.model_dir,name)
        try:
            print('Loading checkpoint %s' % ckp_path)
            ckp = torch.load(ckp_path, map_location='cuda:0')
        except FileNotFoundError:
            print('No checkpoint %s' % ckp_path)
            return

        self.net.load_state_dict(ckp['net'])

        self.initial_epoch = ckp['initial_epoch'] + 1
        for i in range(1,self.initial_epoch):
            self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:%f"%(self.lr))
        print('------------------------------------------------------------------------------')

        print('Continue Train after %d round' % self.initial_epoch)

    def save_checkpoint_net(self,name,epoch):
        ckp_path = os.path.join(self.model_dir,name)
        obj = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'initial_epoch': epoch,
        }
        torch.save(obj,ckp_path)


    def inf_batch(self,batch):
        X, Y = batch['X'].to(self.device), batch['Y'].to(self.device)

        g_derain,derain,refine = self.net(X)

        self.optimizer.zero_grad()

        """backward"""
        g_Y = F.interpolate(Y, size=(Y.shape[2] // 4, Y.shape[3] // 4), mode='bilinear', align_corners=True)
        l1_loss1 = self.l1_loss(g_derain, g_Y)
        l1_loss2 = self.l1_loss(derain, Y)
        l1_loss3 = self.l1_loss(refine, Y)
        loss = l1_loss1 + l1_loss2 + l1_loss3

        loss.backward()
        self.optimizer.step()

        ssim = self.ssim_loss(derain, Y)
        return loss.data.cpu().numpy(),ssim.data.cpu().numpy()


    def inf_batch_val(self,batch):
        X,Y = batch['X'].to(self.device),batch['Y'].to(self.device)
        padh = int(batch['padh'])
        padw = int(batch['padw'])

        with torch.no_grad():
            # g_derain,derain = self.net(X)
            _,_,derain = self.net(X)

            b, c, h, w = derain.shape
            derain = derain[: , : , 0:h - padh , 0:w - padw]

        ssim = self.ssim_loss(derain,Y)
        psnr = PSNR(derain.data.cpu().numpy()*255,Y.data.cpu().numpy()*255)
        return ssim.data.cpu(),psnr

def run_train_val():
    sess = Session()
    sess.load_checkpoint_net(cfg.ckp_name)
    dt_val = sess.get_val_dataloader()
    best_epoch = 0
    best_psnr = 0
    best_ssim = 0
    tags = ['Train_L1','Train_SSIM','Eval_PSNR','Eval_SSIM','lr']

    # for epoch in range(sess.initial_epoch, sess.n_epochs):
    for epoch in range(sess.initial_epoch, sess.n_epochs+1):
        dt_train = sess.get_train_dataloader(epoch)
        # print(dt_train.batch_size)
        time_start = time.time()
        l1_all = 0
        SSIM_all = 0
        train_sample = 0
        sess.net.train()
        for idx_iter,batch_train in enumerate(tqdm(dt_train),0):

            l1_loss,ssim=sess.inf_batch(batch_train)
            l1_all += l1_loss
            SSIM_all += ssim

            train_sample += 1
        SSIM = SSIM_all/train_sample
        l1_all  = l1_all * dt_train.batch_size


        """Evaluation"""

        if epoch % cfg.val_freq == 0:           #5个epoch保存一次
            ssim_val = []
            psnr_val = []
            sess.net.eval()
            #sess.save_checkpoint_net('net_%s_epoch' % str(epoch+1),epoch+1)
            for i,batch_val in enumerate(tqdm(dt_val),0):
                ssim, psnr = sess.inf_batch_val(batch_val)
                ssim_val.append(ssim)
                psnr_val.append(psnr)


            ssim_val = torch.stack(ssim_val).mean().item()
            psnr_val = np.stack(psnr_val).mean().item()

            sess.tb_writer.add_scalar(tags[2],psnr_val,epoch)
            sess.tb_writer.add_scalar(tags[3],ssim_val,epoch)

            logfile = open('./log_test/' + 'val_RealRain1K_H_Baseline' + '.txt', 'a+')
            logfile.write(
                'epoch = ' + str(epoch) + '\t'
                 'ssim  = ' + str(ssim_val) + '\t'
                 'pnsr  = ' + str(psnr_val) + '\t'
                 '\n\n'
            )



            #如果psnr_avg大于best_psnr则单独保存
            if(psnr_val >= best_psnr):
                best_psnr = psnr_val
                best_ssim = ssim_val
                best_epoch = epoch

                sess.save_checkpoint_net('best',epoch)
            print("[epoch %d PSNR: %.4f SSIM:%.4f --- best_epoch %d Best_PSNR %.4f Best_SSIM %.4f]" % (epoch, psnr_val,ssim_val ,best_epoch, best_psnr,best_ssim))
        sess.scheduler.step()

        '''calculation time'''
        print("------------------------------------------------------------------")

        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tSSIM: {:.4f}\tBase_lr {:.8f}".format(epoch,time.time() - time_start,l1_all, SSIM,sess.scheduler.get_last_lr()[0]))


        print("------------------------------------------------------------------")
        sess.save_checkpoint_net('model_latest.pth',epoch)
        sess.tb_writer.add_scalar(tags[0], l1_all, epoch)
        sess.tb_writer.add_scalar(tags[1], SSIM, epoch)
        sess.tb_writer.add_scalar(tags[4], sess.optimizer.param_groups[0]['lr'], epoch)


if __name__ == '__main__':
    run_train_val()