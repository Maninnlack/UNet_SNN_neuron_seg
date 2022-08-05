import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import ImageDataset
from Unet_INN import UNet_INN
from Unet_ANN import UNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device('cuda"0' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device.')

test_img_path = './data/test'
test_mask_path = './data/test_label'

model_ANN_path = './model_save/ANN_UNet_data.pth'
model_INN_path = './model_save/INN_UNet_data.pth'

def test_func(test_img_path, test_mask_path, model_ANN_path, model_INN_path):
    test_dataset = ImageDataset(test_img_path, test_mask_path, dtype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    