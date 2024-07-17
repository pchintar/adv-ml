import os
from time import time
import numpy as np
import pandas as pd
import torch
import random
import utils
import math
import model_arch as ma
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import pathlib
from simba import SimBA
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

image_size = 32
attacker = SimBA(model, 'gtsrb', image_size)

num_runs = len(train_dataset) # number of image samples
num_iters = 20  # max number of iterations
batch_size = 64

batch_file = '%s/images_%s_%d.pth' % ('gg', model, num_runs)


images = torch.zeros(num_runs, 3, image_size, image_size)
labels = torch.zeros(num_runs).long()
preds = labels+1
while preds.ne(labels).sum() > 0:
    idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
    for i in list(idx):
        print('gg ' + str(i))
        images[i], labels[i] = train_dataset[random.randint(0, len(train_dataset) - 1)]
    preds[idx], _ = utils.get_preds(model, images[idx], 'gtsrb', batch_size=batch_size)

    torch.save({'images': images, 'labels': labels}, batch_file)

n_dims = 3 * image_size * image_size
max_iters = num_iters

N = int(math.floor(float(num_runs) / float(batch_size)))

freq_dims = 32
stride = 7
epsilon = 0.2
linf_bound = 0.0
targeted = 'store_true'
order = 'rand'
pixel_attack = 'store_true'
log_every = 1
for i in range(N):
    print(i)
    upper = min((1+i) * batch_size, num_runs)
    images_batch= images[i * batch_size : upper]
    labels_batch = labels[i * batch_size : upper]
    
    adv, probs, succs, queries, l2_norms, linf_norms = attacker.simba_batch(
        images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, pixel_attack=pixel_attack, log_every=log_every)

    if i == 0:
        all_adv = adv
        all_probs = probs
        all_succs = succs
        all_queries = queries
        all_l2_norms = l2_norms
        all_linf_norms = linf_norms
    else:
        all_adv = torch.cat([all_adv, adv], dim=0)
        all_probs = torch.cat([all_probs, probs], dim=0)
        all_succs = torch.cat([all_succs, succs], dim=0)
        all_queries = torch.cat([all_queries, queries], dim=0)
        all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
        all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
    
    prefix = 'pixel'

    savefile = '%s/%s_%s_%d_%d_%d_%.4f_%s%s.pth' % (
        DEST_DIR, prefix, model, num_runs, num_iters, freq_dims, epsilon, order, '')
    
    torch.save({'adv': all_adv, 'probs': all_probs, 'succs': all_succs, 'queries': all_queries,
                'l2_norms': all_l2_norms, 'linf_norms': all_linf_norms}, savefile)

print('dunzo')