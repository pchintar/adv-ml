import os
import torch
import torchvision.transforms as tt
from PIL import Image
import pathlib
import pandas as pd
from resnet import arch
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

    # Data
DATA_DIR = '../../Dataset/gtsrb/'
# MODEL_PATH = '../../Models/saved_models/saved_model_gtsrb.pth'
MODEL_PATH = '../../Models/saved_models/saved_model_gtsrb_retrained_simba.pth'
classes = os.listdir(DATA_DIR + 'Train/')

RESULT_DIR = '../../Results/'

TEST_DIR = DATA_DIR + 'Test/'
TEST_DIR = pathlib.Path(TEST_DIR)
test_images = sorted(list(TEST_DIR.glob('*')))

TARGET_DF_DIR = DATA_DIR + 'Test.csv'
target_df = pd.read_csv(TARGET_DF_DIR)
target_df = target_df[['ClassId']].copy()


print(classes)

    # Data transforms (normalization & data augmentation)
stats = ((0.3403, 0.3122, 0.3214), (0.2751, 0.2642, 0.2706))
test_tfms = tt.Compose([tt.Resize(size=(32, 32)), tt.ToTensor(), tt.Normalize(*stats,inplace=True)])

    # Data Loading to Device
device = arch.get_default_device()
model = arch.to_device(arch.Resnet9(3, 43), device)
model.load_state_dict(torch.load(MODEL_PATH))

    # predict on test set
def predict_image(test_images, model):
    predictions = []
    count = 0
    for img in test_images:
        print(count)
        count += 1
        if len(str(img)) != len(str(test_images[0])):
            continue
        image = Image.open(str(img))
        image = test_tfms(image)
        image.unsqueeze_(0)
        # Convert to a batch of 1
        xb = arch.to_device(image, device)
        # Get predictions from model
        yb = model(xb)
        # Pick index with highest probability
        _, preds  = torch.max(yb, dim=1)
        # Retrieve the class label
        predictions.append(classes[preds[0].item()])
    
    return predictions

pp = predict_image(test_images, model)

prediction_df = pd.DataFrame(pp, columns=['Predictions'])
# prediction_df.to_csv('gtsrb_adv_test_predictions.csv')
# prediction_df = pd.read_csv(RESULT_DIR + 'gtsrb_test_predictions.csv')

true_count = 0

for i in range(0, len(prediction_df)):
    if str(prediction_df['Predictions'][i]) == str(target_df['ClassId'][i]):
        true_count += 1

print('Test accuracy:' + str(true_count/len(prediction_df)) + ' total count = ' + str(len(target_df)))

report = metrics.confusion_matrix(target_df['ClassId'].astype(str), prediction_df['Predictions'].astype(str))
print(report)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(RESULT_DIR + 'gtsrb_adv_test_report_simba.csv')

print(precision_recall_fscore_support(target_df['ClassId'].astype(str), prediction_df['Predictions'].astype(str), average='macro'))
print(precision_recall_fscore_support(target_df['ClassId'].astype(str), prediction_df['Predictions'].astype(str), average='micro'))
print(precision_recall_fscore_support(target_df['ClassId'].astype(str), prediction_df['Predictions'].astype(str), average='weighted'))