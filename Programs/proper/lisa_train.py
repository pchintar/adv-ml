import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as tt
from resnet import arch

    #Data
data_dir = '../../Dataset/lisa/Data_reduced'
classes = os.listdir(data_dir)
print(classes)

    # Transformations

stats = ((0.4773, 0.4704, 0.4808), (0.2578, 0.2574, 0.2782))      

pre_tfms = tt.Compose([tt.Resize(size=(64, 64)), tt.ToTensor()])
train_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])   

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
        print(num_batches)
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


    # Train Data preparation

train_dataset = ImageFolder(root=data_dir, transform=train_tfms)
valid_dataset = ImageFolder(root=data_dir, transform=valid_tfms)

num_train = len(train_dataset)
indices = list(range(num_train))
val_size = int(np.floor(0.1 * num_train))
test_size = val_size
train_size = num_train - (2 * val_size)

train_ds, val_ds, test_ds = random_split(train_dataset, [train_size, val_size, val_size], generator=torch.Generator().manual_seed(42))

batch_size = 16

train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, pin_memory=True)
valid_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=True)
test_ds = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=True)



    #  Data Loading to Device

device = arch.get_default_device()
print(device)

train_dl = arch.DeviceDataLoader(train_dl, device)
valid_dl = arch.DeviceDataLoader(valid_dl, device)

    # Model to Device

model = arch.to_device(arch.Resnet9(3, 18), device)
print(model)

    # Training Setup

epochs = 16
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

    #training

history = arch.fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)

print(history)

torch.save(model.state_dict(), '../../Models/saved_models/saved_model_lisa_reduced_2.pth' )