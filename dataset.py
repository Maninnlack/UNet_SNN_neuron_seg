import os
import torch
import random
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, img_path, mask_path, dtype='test', times=1):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.times = times    # 数据扩增倍数
        self.dtype = dtype
        if dtype == 'train':
            num = 500
        elif dtype == 'val':
            num = 50
        else:
            num = 1

        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)
        if num < len(self.img_list):
            self.img_list = random.sample(self.img_list, num)

        self.img_path_list = [os.path.join(self.img_path, i) for i in self.img_list]
        self.mask_path_list = [os.path.join(self.mask_path, i) for i in self.mask_list]
        

    def __len__(self):
        return len(self.img_list) * self.times

    def __getitem__(self, index):
        if self.times == 1:
            img = Image.open(self.img_path_list[index]).convert('RGB')
            mask = Image.open(self.mask_path_list[index]).convert('L')
        else:
            img = Image.open(self.img_path_list[int(index/self.times)]).convert('RGB')
            mask = Image.open(self.mask_path_list[int(index/self.times)]).convert('L')
            angle = transforms.RandomRotation.get_params([-180,180])    # 旋转角度
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])
        img = transformer(img)
        mask = transformer(mask)

        if self.dtype == 'test': 
            print('test with ' + self.img_path_list[index])

        return {'img': img,
                'mask': mask}
                

if __name__ == '__main__':
    train_img_path = './data/train'
    train_mask_path = './data/train_label'
    val_img_path = './data/val'
    val_mask_path = './data/val_label'
    test_img_path = './data/test'
    # test_mask_path = './data/test_mask'

    train_dataset = ImageDataset(train_img_path, train_mask_path, dtype='train', times=10)
    val_dataset = ImageDataset(val_img_path, val_mask_path, dtype='val')
    # test_dataset = ImageDataset(test_img_path, test_mask_path, dtype='test')
    print(f"read {len(train_dataset)} training data!")
    print(f"read {len(val_dataset)} validation data!")