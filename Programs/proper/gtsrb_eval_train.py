import os
import torch
import numpy as np
import torchvision.transforms as tt
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
from resnet import arch
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

    # Data
DATA_DIR = '../../Dataset/gtsrb/Train'
# DATA_DIR = '../../Attack_images/dpfl'
# MODEL_PATH = '../../Models/saved_models/saved_model_gtsrb.pth'
MODEL_PATH = '../../Models/saved_models/saved_model_gtsrb_retrained_simba.pth'
classes = os.listdir(DATA_DIR)
REPORT_NAME = 'gtsrb_pure_train_report_simba.csv'
PREDICTION_PATH = 'gtsrb_pure_train_predictions_retrained_simba.csv'

REPORT_DIR = '../../Results/'

print(classes)

    # Data transforms (normalization & data augmentation)
stats = ((0.3403, 0.3122, 0.3214), (0.2751, 0.2642, 0.2706))
train_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])
train_dataset = ImageFolder(root=DATA_DIR, transform=train_tfms)
batch_size = 1
train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    # Data Loading to Device
device = arch.get_default_device()
model = arch.to_device(arch.Resnet9(3, 43), device)
model.load_state_dict(torch.load(MODEL_PATH))

train_dl = arch.DeviceDataLoader(train_dl, device)

    # Predict on test set
def predict_image(test_dl, model, device):
    pp = []
    count = 0
    tt = []
    for img, label in test_dl:
        count += 1
        print(count)
        xb = arch.to_device(img, device)
        yb = model(xb)
        _, preds = torch.max(yb, dim=1)
        pp.append(preds[0].item())
        tt.append(label[0].item())
    return [pp, tt]

pred_values = predict_image(train_dl, model, device)

prediction_df = pd.DataFrame(pred_values[0], columns=['Predictions'])
target_df = pd.DataFrame(pred_values[1], columns=['ClassId'])

prediction_df.to_csv(PREDICTION_PATH)
# target_df.to_csv('gtsrb_train_targets.csv')

# pred_df = pd.read_csv('gtsrb_train_predictions.csv')
# target_df = pd.read_csv('gtsrb_train_targets.csv')

true_count = 0

for i in range(0, len(prediction_df)):
    if str(prediction_df['Predictions'][i]) == str(target_df['ClassId'][i]):
        true_count += 1

print('Test accuracy:' + str(true_count/len(prediction_df)) + ' total count = ' + str(len(target_df)))

report = metrics.confusion_matrix(target_df['ClassId'], prediction_df['Predictions'])
print(report)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORT_DIR + REPORT_NAME)

print(precision_recall_fscore_support(target_df['ClassId'], prediction_df['Predictions'], average='macro'))
print(precision_recall_fscore_support(target_df['ClassId'], prediction_df['Predictions'], average='micro'))
print(precision_recall_fscore_support(target_df['ClassId'], prediction_df['Predictions'], average='weighted'))
