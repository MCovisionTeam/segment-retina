import skimage
import numpy as np 
import cv2
import pandas as pd 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import torch
import os
import albumentations as A 
from collections import Counter
from sklearn.model_selection import train_test_split
import skimage.io
import torchvision
import segmentation_models_pytorch as smp
import imageio
from glob import glob

class RetinaDataset(torch.utils.data.Dataset):

    def __init__(self, images, masks, augmentations):
        self.images = images
        self.masks = masks
        self.augmentations = augmentations 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        # uint8 for data aug
        image = np.array(skimage.io.imread(image_path)).astype(np.uint8)
        mask = np.array(skimage.io.imread(mask_path)).astype(np.uint8)

        data = self.augmentations(image=image, mask=mask)
        image = data['image']
        mask = data['mask']

        image = np.array(image).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)

        image = image / 255
        image = torch.Tensor(image)
        image = torch.permute(image, (2,0,1))

        mask = np.where(mask>0, 255, 0)
        mask = mask / 255
        mask = torch.Tensor(mask)
        
        return image, mask, image_path


def plot_pred(loader,model, epoch, working_directory, experiment_name):

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(working_directory, experiment_name, 'animations'),exist_ok=True)

    for images, masks, image_path in tqdm(loader):
        
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        masks = torch.unsqueeze(masks, 1)
        out = model(images)
        pred_mask = out[0,:,:,:]
        pred_mask = (pred_mask>0.5)*1.0
        res = pred_mask.detach().cpu().numpy()
        res = res.transpose((1,2,0))
        # to rgb
        res = np.repeat(res,3,axis=2)
        plt.figure(figsize=(7,7))
        plt.imshow(res)
        plt.axis('off')
        plt.savefig(os.path.join(working_directory, experiment_name, 'animations/') + str(epoch) +'.png')
        plt.show()
        break
  

def create_gif(folder):
    list_path_masks = sorted(glob.glob(folder+'/*.png'))
    images = [imageio.imread(path_mask) for path_mask in list_path_masks]
    imageio.mimsave("animation.gif",images,duration=0.5)