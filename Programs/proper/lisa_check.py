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
data_dir = '../../Dataset/lisa/Data'
classes = os.listdir(data_dir)
print(classes)

pre_tfms = tt.Compose([tt.ToTensor()])
train_dataset = ImageFolder(root=data_dir, transform=pre_tfms)

train_dl = DataLoader(train_dataset, batch_size=1, num_workers=0, pin_memory=True)

name = []
freq = []

prev_class = -1
count = 0
for img, label in train_dl:
    t = label[0].item()
    if prev_class != t:
        print(t)
        if prev_class != -1:
            name.append(classes[prev_class])
            freq.append(count)
        count = 1
        prev_class = t
    else:
        count += 1 


pp = sorted(zip(freq, name))
print(pp)
# number_to_name = {}

# i = 0
# for cls in classes:
#     number_to_name.update({i : classes[i]})
#     i += 1

# print(number_to_name)
# prev_class = -1
# sizes = {}
# count = []
# counter = -1
# for img, label in train_dl:
#     t = label[0].item()
#     if prev_class != t:
#         print(t)
#         if prev_class != -1:
#             sizes.update({counter : number_to_name[prev_class]})
#             count.append(counter)
#         counter = 1
#         prev_class = t
#     else:
#         counter += 1

# print(count)

# print(sizes)

# count.sort(reverse=True)

# for c in count:
#     print(sizes[c], end=":  ")
#     print(c)
