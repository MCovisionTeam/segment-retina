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


def show_images(images, labels, preds, images_path):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(32, 16))
    
    for i, (image, mask, image_path) in enumerate(zip(images, labels, images_path)):
        orig_image = skimage.io.imread(image_path)
        width, height = orig_image.shape[:2]
        image = image.numpy().transpose((1, 2, 0))
        image = cv2.resize(image, dsize=(height,width), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask.numpy(),2)
        mask = np.repeat(mask,3, axis=2)
        mask = cv2.resize(mask, dsize=(height,width), interpolation=cv2.INTER_NEAREST)

        axs[0, i].imshow(image)
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        
        axs[1, i].imshow(mask)
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def show_preds(dloader_test, model):

    # fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(32, 16))
    
    for i, (image, mask, image_path) in enumerate(dloader_test):

        image_path = image_path[0]
        orig_image = skimage.io.imread(image_path)
        print(orig_image.shape)

        output = model(image.to(DEVICE)) 
        output = (output>0.5)*1.0
        output = output.squeeze(0).detach().cpu().numpy()
        output = output[0,0,:,:]
        output = cv2.resize(output, dsize=(orig_image.shape[1],orig_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        output = np.expand_dims(output,2)
        output = np.repeat(output,3, axis=2)

        mask = np.array(mask[0,:,:])
        mask = cv2.resize(mask, dsize=(orig_image.shape[1],orig_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask,2)
        mask = np.repeat(mask,3, axis=2)


        image = image[0,:,:,:]
        image = image.numpy().transpose((1, 2, 0))
        image = cv2.resize(image, dsize=(orig_image.shape[1],orig_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        plt.figure(figsize=(15,15))
        plt.imshow(image)
        plt.title(os.path.basename(image_path))
        plt.axis('off')

        plt.figure(figsize=(15,15))
        plt.imshow(mask)
        plt.title('GROUND TRUTH')
        plt.axis('off')

        plt.figure(figsize=(15,15))
        plt.imshow(output)
        plt.title('PREDICTION')
        plt.axis('off')

        break
            