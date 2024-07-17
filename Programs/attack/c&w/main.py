import os
from time import time
import numpy as np
import pandas as pd
import torch
import model_arch as ma
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as tt
import pathlib
from torchvision.utils import save_image
    #DATA LOADING
DATA_DIR = '../../../Dataset/gtsrb/'
MODEL_PATH = '../../../Models/resnet50models_v2/saved_model_premium.pth'
TEST_DIR = '../../../Dataset/gtsrb/Test/'
TEST_DIR = pathlib.Path(TEST_DIR)
TEST_LABELS = '../../../Dataset/gtsrb/Test.csv'
DEST_DIR = '../../../Attack_images/cw/'
classes = os.listdir(DATA_DIR + '/Train')
print(classes)

    # TRANSFORMAITONS
stats = ((0.3403, 0.3122, 0.3214), (0.2751, 0.2642, 0.2706))
train_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])

train_dataset = ImageFolder(root=DATA_DIR + '/Train', transform=train_tfms)
batch_size = 1
train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    # C W attack

def cw_l2_attack(model, images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01, device) :

    # Define f-function
    def f(x) :

        outputs = model(x)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.byte())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)
        
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
    
    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10
    
    for step in range(max_iter) :

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1)

    return attack_images

def run_attack(model, data_loader, device):
    correct = 0
    total = 0

    for images, labels in data_loader:
        
        images = cw_l2_attack(model, images, labels, targeted=False, c=0.1, device=device)
        outputs = model(images)
        
        _, pre = torch.max(outputs.data, 1)

        total += 1
        correct += (pre == labels).sum()

        save_imgs = images.cpu().detach().numpy()

        for img, output, label in zip(save_imgs, outputs, labels):
            if label.item() != output.item():
                exist = os.path.exists(DEST_DIR + str(classes[label.item()]))
                if not exist:
                    os.makedirs(DEST_DIR + str(classes[label.item()]))
                save_image(img, DEST_DIR + str(classes[label.item()]) + '/' + str(total) + '.png')
     
    print('Accuracy of test text: %f %%' % (100 * float(correct) / total))

    # Load, dataset

device = ma.get_default_device()
model = ma.to_device(ma.Resnet9(3, 43), device)
model.load_state_dict(torch.load(MODEL_PATH))

train_dl = ma.DeviceDataLoader(train_dl, device)

model.eval()

run_attack(model, train_dl, device)


