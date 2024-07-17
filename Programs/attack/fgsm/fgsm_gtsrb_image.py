import os
from time import time
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
from torchvision.utils import save_image
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

    #DATA LOADING
DATA_DIR = '../../Dataset/gtsrb/'
MODEL_PATH = '../../Models/resnet50models_v2/saved_model_premium.pth'
TEST_DIR = '../../Dataset/gtsrb/Test/'
TEST_DIR = pathlib.Path(TEST_DIR)
TEST_LABELS = '../../Dataset/gtsrb/Test.csv'
classes = os.listdir(DATA_DIR + 'Train')
DEST_DIR = '../../Attack_images/fgsm/'
print(classes)

    # TRANSFORMAITONS
stats = ((0.3403, 0.3122, 0.3214), (0.2751, 0.2642, 0.2706))
train_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])
train_dataset = ImageFolder(root=DATA_DIR + '/Train', transform=train_tfms)

batch_size = 1

train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    #DEVICE SETUP

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print("device: CUDA")
        return torch.device('cuda')
    else:
        print("device: CPU")
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    #MODEL

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Resnet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

    #FGSM

def fgsm_attack(image, epsilon, data_grad):
    """Collect the element-wise sign of the data gradient"""
    sign_data_grad = data_grad.sign()
    """Create the perturbed image by adjusting each pixel of the input image"""
    perturbed_image = image + epsilon*sign_data_grad
    """Adding clipping to maintain [0,1] range"""
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    """Return the perturbed image"""
    return perturbed_image

def fgsm_test(model, train_loader, epsilon):

    # Accuracy counter
    correct = 0
    counter = 0

    for batch in train_loader:
        counter += 1
        data, target = batch
        print(counter, end=': ')

        """Set requires_grad attribute of tensor. Important for Attack"""
        data.requires_grad = True

        """Forward pass the data through the model"""
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        _, init_pred = torch.max(output, dim=1)
        # init_pred = classes[init_pred[0].item()]
        init_pred = init_pred[0].item()

        print("t:", end=' ')
        print(target.item(), end=' ')
        print("i:", end=' ')
        print(init_pred, end=' ')

        """If the initial prediction is wrong, dont bother attacking, just move on"""
        if str(init_pred) != str(target.item()):
            print('\n')
            continue

        """Calculate the loss"""
        loss = F.nll_loss(output, target)

        """Zero all existing gradients"""
        model.zero_grad()

        """Calculate gradients of model in backward pass"""
        loss.backward()

        """Collect datagrad"""
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        _, final_pred = torch.max(output, dim=1)
        # final_pred = classes[final_pred[0].item()]
        final_pred = final_pred[0].item()
        print("f:", end=" ")
        print(final_pred)
        
        if str(final_pred) == str(target.item()):
            correct += 1
        else:
            # Save some adv examples for visualization later
            # adv_ex = perturbed_data.squeeze().detach().cpu()
            # adv_examples.append((init_pred, adv_ex))
            exist = os.path.exists(DEST_DIR + 'epsilon_' + str(epsilon * 10) + '/' + str(classes[target.item()]))
            if not exist:
                os.makedirs(DEST_DIR + 'epsilon_' + str(epsilon * 10) + '/' + str(classes[target.item()]))
            # save_image(adv_ex, DEST_DIR + 'epsilon_' + str(epsilon * 10) + '/' + str(classes[target.item()]) + '/' + str(counter) + '.png')

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(train_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(train_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc

    # Load, setup for FGSM

device = get_default_device()
model = to_device(Resnet9(3, 43), device)
model.load_state_dict(torch.load(MODEL_PATH))

model.eval()

train_dl = DeviceDataLoader(train_dl, device)

epsilons = [0, 0.05, 0.1, 0.15, .2, .4]

accuracies = []
examples = []

print(len(train_dl))

for eps in epsilons:
    acc= fgsm_test(model, train_dl, eps)
    accuracies.append(acc)
print(accuracies)