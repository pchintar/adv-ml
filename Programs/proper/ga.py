import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import SequentialSampler
import torchvision.transforms as tt
from resnet import arch

data_dir = '../../Dataset/gtsrb/'
classes = os.listdir(data_dir + '/Train')

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

train_seq_sampler = SequentialSampler(train_idx)
valid_seq_sampler = SequentialSampler(valid_idx)

batch_size = 32

train_dl = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler,
    num_workers=0, pin_memory=True)

valid_dl = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler,
    num_workers=0, pin_memory=True)

train_seq_dl = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, sampler=train_seq_sampler,
    num_workers=0, pin_memory=True)

valid_seq_dl = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, sampler=valid_seq_sampler,
    num_workers=0, pin_memory=True)

device = arch.get_default_device()
print(device)

train_dl = arch.DeviceDataLoader(train_dl, device)
valid_dl = arch.DeviceDataLoader(valid_dl, device)

train_seq_dl = arch.DeviceDataLoader(train_seq_dl, device)
valid_seq_dl = arch.DeviceDataLoader(valid_seq_dl, device)

model = arch.to_device(arch.Resnet9(3, 43), device)
# print(model)

i = 0
for batch in train_dl:
    print(batch[1])
    i += 1
    if i == 500:
        break

def train_model(model, train_dl):
    epochs = 5
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    history = arch.fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)

    torch.save(model.state_dict(), '../../Models/saved_models/saved_model_gtsrb_sa.pth' )

    return model

def fitness(train_dl, alpha_dl, model):
    # metrics is recall
    for train_sample, alpha in zip(train_dl, alpha_dl):
        image, label = train_sample
        out = model(image + alpha)
    

def two_player_ga(train_dl):
    model = train_model(model, train_dl)
    max_payoff = 0
    exit_loop = False
    population = alpha_dl
    F_X_train = fitness(train_dl, alpha_dl)
