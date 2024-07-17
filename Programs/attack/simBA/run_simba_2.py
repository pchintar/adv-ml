import os
from time import time
import numpy as np
import pandas as pd
import torch
import model_arch as ma
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import pathlib
from simba_2 import SimBA
from torchvision.utils import save_image
    #DATA LOADING
DATA_DIR = '../../../Dataset/gtsrb/'
MODEL_PATH = '../../../Models/resnet50models_v2/saved_model_premium.pth'
TEST_DIR = '../../../Dataset/gtsrb/Test/'
TEST_DIR = pathlib.Path(TEST_DIR)
TEST_LABELS = '../../../Dataset/gtsrb/Test.csv'
DEST_DIR = '../../../Attack_images/simba/'
classes = os.listdir(DATA_DIR + '/Train')
print(classes)

    # TRANSFORMAITONS
stats = ((0.3403, 0.3122, 0.3214), (0.2751, 0.2642, 0.2706))
train_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])

train_dataset = ImageFolder(root=DATA_DIR + '/Train', transform=train_tfms)
batch_size = 1
train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    # Load, dataset

device = ma.get_default_device()
model = ma.to_device(ma.Resnet9(3, 43), device)
model.load_state_dict(torch.load(MODEL_PATH))

train_dl = ma.DeviceDataLoader(train_dl, device)

model.eval()

attacker = SimBA(model, device)

def run_attack(model, train_loader, attacker):
    counter  = 0
    epsilon = 0.1
    max_it = 1000
    i = 0
    for batch in train_loader:
        print(i, end=': ')
        i += 1
        data, target = batch
        
        # run 16th again
        if target.item() < 25:
            continue

        x_adv = attacker.attack(data, epsilon, max_it)
        output = model(x_adv)

        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        _, final_pred = torch.max(output, dim=1)
        final_pred = final_pred[0].item()
        print(final_pred)
        print(target.item())
        if str(target.item()) != str(final_pred):      # the attack worked
            counter += 1
            exist = os.path.exists(DEST_DIR + str(classes[target.item()]))
            if not exist:
                os.makedirs(DEST_DIR + str(classes[target.item()]))
            save_image(x_adv, DEST_DIR + str(classes[target.item()]) + '/' + str(counter) + '.png')
        
    return counter


result = run_attack(model, train_dl, attacker)
print(result)
print('dunzo')