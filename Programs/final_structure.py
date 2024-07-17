#import matplotlib.pyplot as plt
import numpy as np
import os
# import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import pathlib
import cv2

	# Data Read

#data_dir = '../Dataset_GTSRB/archive/Train'
data_dir = '../Dataset/gtsrb/Train'
data_dir = pathlib.Path(data_dir)
classes = os.listdir(data_dir)

print("DATA READ DONE")

	# Training and Validation Split

img_height, img_width = 100, 100
batch_size = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode="categorical",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_name = train_ds.class_names

print('TRAIN-VALIDATION SPLIT DONE')

	# Model

resnet_model = Sequential()

pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(100,100,3),
                                                  pooling='avg',classes=43,
                                                  weights='imagenet')
for layer in pretrained_model.layers:
  layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(43, activation='softmax'))

resnet_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print('MODEL CREATION DONE')

	# Training

epoch = 10

resnet_50_gtsrb = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch
)

print('TRAINING DONE')

 	# Saving Model

print("Doing Write Op")
resnet_50_gtsrb.save('../Models/resnet_50_gtsrb')

print('SAVING DONE')


	#Loading Model

# resnet_50_gtsrb = tf.keras.models.load_model('../Models/resnet_50_gtsrb')

# print("LOADING COMPLETE")

	# Testing

#test_dir = '../Dataset_GTSRB/archive/Test'
test_dir = '../Dataset/gtsrb/Test'
test_dir = pathlib.Path(test_dir)

# #print(test_dir)

test_images = sorted(list(test_dir.glob('*')))	# the test images contains repeats

#print(len(test_images))

#true1_df = pd.read_csv('../Dataset_GTSRB/archive/Test.csv')
true1_df = pd.read_csv('../Dataset/gtsrb/Test.csv')
true_df = true1_df[['ClassId']].copy()

predictions = []
count = 0

for img in test_images:
	count += 1
	print(count)
	if len(str(img)) != len(str(test_images[0])):	# repeats filter
		continue
	image = cv2.imread(str(img))
	image_resized = cv2.resize(image, (img_height, img_width))
	image = np.expand_dims(image_resized, axis=0)
	pred = resnet_50_gtsrb.predict(image)
	output_class = class_name[np.argmax(pred)]
	predictions.append(output_class)

pred_df = pd.DataFrame(predictions, columns=['Predictions'])
pred_df.to_csv('../Prediction_results/file1.csv')

# pred_df = pd.read_csv('../Prediction_results/file1.csv')

print('PREDICTIONS DONE')

# print('pred_df ka size: ' + str(pred_df.shape[0]))
# print('test_df ka size: ' + str(true_df.shape[0]))

true_count = 0

for i in range(0, 12630):
	if pred_df['Predictions'][i] == true_df['ClassId'][i]:
		true_count += 1

print('ACCURACY CALCULATED')
print(str(true_count/12630))

f = open("../Prediction_results/file_acc.txt", "a+")
f.write('the accuracy is: ' + str(true_count / 12630) + '\n')
f.close()

print("DONE")