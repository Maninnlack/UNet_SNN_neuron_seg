import argparse
import os
import tqdm
import time
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch import optim
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Unet_ANN import UNet
from utils.loss import FocalLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# image and mask path
train_img_path = './data/train'
train_mask_path = './data/train_label'
val_img_path = './data/val'
val_mask_path = './data/val_label'

checkpoint_path = './model_save'

parse = argparse.ArgumentParser(description='UNet-INN')
parse.add_argument('--device', default='cuda:0', help='运行的设备')
parse.add_argument('-b', '--batchsize', default=8, type=int, help='Batch 大小')
parse.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
parse.add_argument('-e', '--epoch', default=10, type=int, help='epoch数量')

arg = parse.parse_args()

#  nvidia configure
device = torch.device(arg.device if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

# hyper perameters
batch_size = arg.batchsize
epochs = arg.epoch
learning_rate = arg.learning_rate

# dataset
train_dataset = ImageDataset(train_img_path, train_mask_path, dtype='train', times=10)
val_dataset = ImageDataset(val_img_path, val_mask_path, dtype='val', times=10)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# net init
model_name = 'ANN_UNet_data'
net = UNet(in_channels=3, out_channels=1)
net.to(device=device)

# optimizer, loss function, learning rate
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = FocalLoss()    
# 损失函数可选 Diceloss， Focalloss 当最后一层没有加sigmoid时，loss函数要选F.binary_cross_entropy_with_logits

batch_num = 0
train_losses, val_losses = [], [] # 记录全程loss 变化
log = []  # 写入txt记录

# tensorboard 可视化
writer = SummaryWriter()
#writer.add_graph(net, torch.rand([batch_size,3,512,512]).to(device))

try:
    # train
    for epoch in range(epochs):
        net.train() 
        for index, batch in enumerate(train_dataloader):
            image = batch['img'].to(device=device)
            mask = batch['mask'].to(device=device)

            assert image.shape[1] == net.in_channels,\
                f'input channels:{net.in_channels} is not equal to image channels:{image.shape[1]}'

            pred_mask = net(image)
            loss = criterion(pred_mask, mask)    # target mask需要long格式
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()
            train_losses.append(train_loss) # item防止爆显存
            current = index * len(image) + 1
            size_train = len(train_dataloader.dataset)
            writer.add_scalar('train loss', train_loss,  batch_num + epoch * size_train)
            print(f'train loss: {train_loss:>7f} [{current:>5d}/{size_train:>5d}]  epoch:{epoch}/{epochs}')
            log.append(f'train loss: {train_loss:>7f} [{current:>5d}/{size_train:>5d}]  epoch:{epoch}/{epochs}\n')

        # validation
        size_val = len(val_dataloader)      # 这里应该是batchnamber，而不是datanumber
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for index, batch in enumerate(val_dataloader):
                image = batch['img'].to(device)
                mask = batch['mask'].to(device)
                pred_mask = net(image)
                val_loss += criterion(pred_mask, mask).item()
        val_loss = val_loss / size_val
        val_losses.append(val_loss)
        writer.add_scalar('val loss', val_loss, epoch)
        batch_num += 1
        
        print(f'val loss: {val_loss:>7f}  epoch:{epoch}/{epochs}')
        log.append(f'val loss: {val_loss:>7f}  epoch:{epoch}/{epochs}\n')

    print(f'minimum train loss: {min(train_losses):>7f} minimum val loss: {min(val_losses):>7f}  epoch:{epoch}/{epochs}')
    log.append(f'minimum train loss: {min(train_losses):>7f} minimum val loss: {min(val_losses):>7f}  epoch:{epoch}/{epochs}\n')

    with open('./ANN_test.txt', 'a+') as f:
        f.write(model_name + '\n')
        for i in range(len(log)):
            f.write(log[i])

    torch.save(net.state_dict(), checkpoint_path + os.sep + model_name + '.pth')
    print(f'model {model_name} saved!')

except KeyboardInterrupt:
    train_time = time.strftime("%y-%m-%d", time.localtime())
    torch.save(net.state_dict(), checkpoint_path + os.sep + model_name + '_' + train_time + '.pth')
    print(f'Interruptted with model {model_name} saved.')

finally:
    # writer.flush()
    writer.close()
