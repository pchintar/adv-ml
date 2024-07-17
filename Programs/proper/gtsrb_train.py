import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as tt
from resnet import arch


    #Data
data_dir = '../../Dataset/gtsrb/'
classes = os.listdir(data_dir + '/Train')
print(classes)

    # Transformations

stats = ((0.3403, 0.3122, 0.3214), (0.2751, 0.2642, 0.2706))
train_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])   


    # Train Data preparation

train_dataset = ImageFolder(root=data_dir + '/Train', transform=train_tfms)
valid_dataset = ImageFolder(root=data_dir + '/Train', transform=valid_tfms)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))

np.random.seed(123)
np.random.shuffle(indices)


train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

batch_size = 128

train_dl = DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler,
    num_workers=0, pin_memory=True)

valid_dl = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler,
    num_workers=0, pin_memory=True)

    #  Data Loading to Device

device = arch.get_default_device()
print(device)

train_dl = arch.DeviceDataLoader(train_dl, device)
valid_dl = arch.DeviceDataLoader(valid_dl, device)

    # Model to Device

model = arch.to_device(arch.Resnet9(3, 43), device)
print(model)

    # Training Setup

epochs = 8
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

# torch.save(model.state_dict(), '../../Models/saved_models/saved_model_gtsrb.pth' )