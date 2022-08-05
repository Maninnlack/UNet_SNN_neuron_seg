import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import ImageDataset
from Unet_INN import UNet_INN

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')
print(f'Using {device} device.')

test_img_path = './data/test'
test_mask_path = './data/test_label'

model_path = './model_save/INN_UNet_data.pth'

def test_func(test_img_path, test_mask_path, model_path):
    predicted_img = []
    test_dataset = ImageDataset(test_img_path, test_mask_path, dtype='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model = UNet_INN(3, 1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for _, batch in enumerate(test_dataloader):
        img = batch['img']
        mask = batch['mask']
        img_g = img.to(device)

        with torch.no_grad():
            pred = model(img_g)
            predicted_img.append(pred)
    pred_show_img = predicted_img[0].squeeze()
    plt.figure(figsize=(10,5))
    plt.subplot(1, 3, 1)
    plt.imshow(img.squeeze(0).permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title('image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('label')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow((pred_show_img > 0.4).float().cpu().numpy(), cmap='gray')
    plt.title('INN segmentation')
    plt.axis('off')
    plt.savefig('./test_INN.jpg')
    plt.show()

if __name__ == '__main__':
    test_func(test_img_path, test_mask_path, model_path)