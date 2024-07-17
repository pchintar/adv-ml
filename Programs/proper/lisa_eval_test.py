import os
import torch
import numpy as np
import torchvision.transforms as tt
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import pandas as pd
from resnet import arch
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

    # Data
DATA_DIR = '../../Dataset/lisa/Data_reduced'
MODEL_PATH = '../../Models/saved_models/saved_model_lisa_reduced_2.pth'
classes = os.listdir(DATA_DIR)

REPORT_DIR = '../../Results/'

print(classes)

    # Data transforms (normalization & data augmentation)
stats = ((0.4773, 0.4704, 0.4808), (0.2578, 0.2574, 0.2782))
train_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])
test_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])

test_dataset = ImageFolder(root=DATA_DIR, transform=train_tfms)

dataset_size = len(test_dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))

np.random.seed(123)
np.random.shuffle(indices)

train_idx, test_idx = indices[split:], indices[:split]
test_sampler = SubsetRandomSampler(test_idx)

batch_size = 1

test_dl = DataLoader(
    test_dataset, batch_size=batch_size, sampler=test_sampler,
    num_workers=0, pin_memory=True)

    # Data Loading to Device
device = arch.get_default_device()
model = arch.to_device(arch.Resnet9(3, 18), device)
model.load_state_dict(torch.load(MODEL_PATH))

test_dl = arch.DeviceDataLoader(test_dl, device)

    # Predict on test set
def predict_image(test_dl, model, device):
    pp = []
    tt = []
    for img, label in test_dl:
        xb = arch.to_device(img, device)
        yb = model(xb)
        _, preds = torch.max(yb, dim=1)
        pp.append(preds[0].item())
        tt.append(label[0].item())
    return [pp, tt]

pred_values = predict_image(test_dl, model, device)

prediction_df = pd.DataFrame(pred_values[0], columns=['Predictions'])
target_df = pd.DataFrame(pred_values[1], columns=['ClassId'])

# pred_df.to_csv('lisa_test_predictions.csv')
# target_df.to_csv('list_test_targets.csv')

# pred_df = pd.read_csv('lisa_test_predictions.csv')
# target_df = pd.read_csv('list_test_targets.csv')

true_count = 0

for i in range(0, len(prediction_df)):
    if str(prediction_df['Predictions'][i]) == str(target_df['ClassId'][i]):
        true_count += 1

print('Test accuracy:' + str(true_count/len(prediction_df)) + ' total count = ' + str(len(target_df)))

report = metrics.confusion_matrix(target_df['ClassId'], prediction_df['Predictions'])
print(report)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORT_DIR + 'lisa_reduced_test_report_2.csv')

print(precision_recall_fscore_support(target_df['ClassId'], prediction_df['Predictions'], average='macro'))
print(precision_recall_fscore_support(target_df['ClassId'], prediction_df['Predictions'], average='micro'))
print(precision_recall_fscore_support(target_df['ClassId'], prediction_df['Predictions'], average='weighted'))
