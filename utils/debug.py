import time

from torch.optim import Adam
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts
from model.network_v13 import network_v13
#from utils.scheduler import CosineScheduler
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    tb_writer = SummaryWriter(log_dir='./scheduler')

    epoch = 80
    lr = 1e-4
    net = network_v13()

    optimizer = Adam(net.parameters(),lr=lr)

    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=epoch + 5,T_mult=1,eta_min=lr*1e-2)

    for i in range(epoch):
        print(i,'--------lr:',optimizer.param_groups[0]['lr'])
        scheduler.step()
        time.sleep(0.05)
        tb_writer.add_scalar('lr',optimizer.param_groups[0]['lr'],i)