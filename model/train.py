import sys
import os 
sys.path.append(os.getcwd())
import skimage
import numpy as np 
import cv2
import pandas as pd 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import torch
import albumentations as A 
import skimage.io
import torchvision
import segmentation_models_pytorch as smp
import argparse
import yaml
from segment_retina.utils import RetinaDataset
import random
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from segment_retina.utils import plot_pred, create_gif


def parse_args(with_exp_name=False, with_config=True):

    # Parse commend line arguments 
    parser = argparse.ArgumentParser()
    if with_config:
        parser.add_argument('config', type=str, help="Path of the config yaml file")
    if with_exp_name:
        parser.add_argument('name', type=str, help="Name of the simulation. A folder './output/<name>' will be created.")
    args = parser.parse_args()
    
    # config file
    if with_config:
        config_path = args.config 
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) 
    # experiment name
    if with_exp_name:
        experiment_name = args.name

    if with_exp_name and with_config:
        return config, experiment_name
    elif with_exp_name:
        return experiment_name
    elif with_config:
        return config
    else:
        raise ValueError('Bad argument values')


def data_preparation(dataset):

    root = '/content/drive/MyDrive/Mcovision events/segmentation vaisseaux rétiniens/data/Kaggle_retina_segmentation'
    exts = ('jpg', 'JPG', 'png', 'PNG', 'tif', 'gif', 'ppm')

    if dataset == 'CHASE_DB1':
        input_data = os.path.join(root, 'CHASE_DB1/Images')
        images = sorted([os.path.join(input_data, fname) for fname in os.listdir(input_data) if fname.endswith(exts) and not fname.startswith(".")])

        target_data = os.path.join(root, 'CHASE_DB1/Masks')
        masks = sorted([os.path.join(target_data, fname) for fname in os.listdir(target_data) if fname.endswith('_2ndHO.png') and not fname.startswith(".")])

    elif  dataset == 'HRF':
        input_data = os.path.join(root, 'HRF/images')
        images = sorted([os.path.join(input_data, fname) for fname in os.listdir(input_data) if fname.endswith(exts) and not fname.startswith(".")])

        target_data = os.path.join(root, 'HRF/manual1')
        masks = sorted([os.path.join(target_data, fname) for fname in os.listdir(target_data) if fname.endswith(exts) and not fname.startswith(".")])

    assert len(images) == len(masks)

    return images, masks


def train_augmentation(IMG_SIZE):

    return A.Compose([
                        # A.OneOf([A.MotionBlur(p=0.3), A.Blur(blur_limit=15, p=0.3), A.GaussNoise(p=0.2)], p=0.5),
                        # A.OneOf([A.CLAHE(p=0.3),A.RandomGamma(p=0.3)],p=0.5),
                        # A.Rotate(limit=30, p=0.3),
                        # A.RandomBrightnessContrast(p=0.3),
                        # A.RandomCrop(p=0.3, height=256, width=256),
                        A.Resize(IMG_SIZE,IMG_SIZE,interpolation=cv2.INTER_NEAREST)
                        ])
   
def test_augmentation(IMG_SIZE):
        
    return A.Compose([
                        A.Resize(IMG_SIZE,IMG_SIZE,interpolation=cv2.INTER_NEAREST),  
                        ])


def load_data(images, masks, config):

    FINAL_SIZE = config['train']['final_size']
    BATCH_SIZE = config['train']['batch_size']
    

    trainset = RetinaDataset(images[:58], masks[:58], train_augmentation(FINAL_SIZE))
    testset = RetinaDataset(images[58:], masks[58:], test_augmentation(FINAL_SIZE))


    dloader_train = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    dloader_test = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

    # To get the same order for plot prediction
    dloader_test_for_plot = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


    return dloader_train, dloader_test, dloader_test_for_plot


def create_model(config):

    MODEL = config['train']['model_name']
    ENCODER = config['train']['encoder']
    ENCODER_WEIGHTS = config['train']['weights']
    LOSS = config['train']['loss']
    LR = config['train']['lr']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if MODEL == 'unet++':

        model = smp.UnetPlusPlus(encoder_name=ENCODER,
                                encoder_weights=ENCODER_WEIGHTS,
                                classes=1,
                                activation=None,
                                in_channels=3)

    elif MODEL == 'fpn':
        print('FPN')

        model = smp.FPN(encoder_name=ENCODER,
                        encoder_weights=ENCODER_WEIGHTS,
                        classes=1,
                        activation=None,
                        in_channels=3)


    model.to(DEVICE)

    if LOSS == 'binarycrossentropy':

        # no logits
        # loss = torch.nn.BCELoss()
        loss = torch.nn.BCEWithLogitsLoss()
    
    elif LOSS == 'dice':

        # no logits 
        loss = smp.losses.DiceLoss(mode='binary', from_logits=False)
        loss.__name__ = "Dice_loss"


    optimizer = torch.optim.Adam(model.parameters(), lr = LR)

    return model, loss, optimizer


def train_fn(data_loader, model, optimizer, loss):
    
    model.train()
    total_loss = 0.0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for images, masks, image_path in tqdm(data_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        masks = torch.unsqueeze(masks, 1)

        optimizer.zero_grad()
        out = model(images)
        l = loss(out, masks)
        l.backward()
        optimizer.step()
        total_loss += l.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader, model, loss):

    model.eval()
    total_loss = 0.0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad(): #pour ne pas tracer le gradient des variables 

        for images, masks, image_path in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            masks = torch.unsqueeze(masks, 1)

            out = model(images)
            l = loss(out, masks)
            

            total_loss += l.item()

    return total_loss / len(data_loader)


def training(config, dloader_train, dloader_test,dloader_test_for_plot, model, optimizer, loss, working_directory, experiment_name):

    EPOCHS = config['train']['epochs']
    best_valid_loss = np.Inf
    training_loss = []
    test_loss = []

    for i in range(EPOCHS):

        train_loss = train_fn(dloader_train, model, optimizer, loss)
        valid_loss = eval_fn(dloader_test,model, loss)
        training_loss.append(train_loss)
        test_loss.append(valid_loss)

        if valid_loss < best_valid_loss:
            torch.save(model, os.path.join(working_directory, experiment_name,"train/best_model.pth"))
            print("SAVED MODEL WITH VALID LOSS : ", valid_loss)
            best_valid_loss = valid_loss
            plot_pred(dloader_test_for_plot,model, i, working_directory, experiment_name)


        print(f"Epoch {i+1} Train loss {train_loss} Valid Loss {valid_loss} ")

    return training_loss, test_loss 


def plot_metrics(working_directory,experiment_name, training_loss, test_loss, config):
 

    num_epochs = config['train']['epochs']
    # save dir
    save_dir = os.path.join(working_directory, experiment_name, 'train', 'figures')
    os.makedirs(save_dir, exist_ok=True)

    # get data to plot
    plt.rcParams.update({'font.size': 25})

    fig = plt.figure(figsize=(15, 10))
    plt.plot(range(num_epochs), training_loss, color="red", label="Training loss")
    plt.plot(range(num_epochs), test_loss, color="blue", label="Test loss")
    plt.ylim(0), plt.legend(), plt.title("Loss")
    plt.savefig(os.path.join(save_dir, 'loss.png'), bbox_inches='tight')



if __name__ == '__main__':

     # parse config
    config, experiment_name = parse_args(with_exp_name=True,with_config=True)
    working_directory = config['train']['working_directory']
    # output directories
    list_of_directories = [working_directory,
                           os.path.join(working_directory, experiment_name),
                           os.path.join(working_directory, experiment_name, 'train'),
                           os.path.join(working_directory, experiment_name, 'val')]

    for directory in list_of_directories:
        os.makedirs(directory, exist_ok=True)

    # save config file
    with open(os.path.join(list_of_directories[1], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    images_chase, masks_chase = data_preparation('CHASE_DB1')
    images_hrf, masks_hrf = data_preparation('HRF')
    images, masks = images_chase + images_hrf, masks_chase + masks_hrf

    # images, masks = data_preparation('HRF')

    # shuffle for robust dataset
    combined = list(zip(images, masks))
    random.seed(42)
    random.shuffle(combined)
    images, masks = zip(*combined)

    print('Len Dataset : {}'.format(len(images)))
    dloader_train, dloader_test, dloader_test_for_plot = load_data(images, masks, config)
    model, loss, optimizer = create_model(config)
    training_loss, test_loss = training(config, dloader_train, dloader_test, dloader_test_for_plot, model, optimizer, loss, working_directory, experiment_name)

    plot_metrics(working_directory,experiment_name, training_loss, test_loss, config)