import pandas as pd
from sklearn.datasets import load_files 

import numpy as np
from glob import glob
import random
import cv2 
from tqdm import tqdm
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50,preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from extract_bottleneck_features import *
from PIL import ImageFile 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout,Flatten
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import SGD

def load_dataset(path):
	data = load_files(path)
	dog_files = np.array(data['filenames'])
	dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
	return dog_files, dog_targets

data_folder_path = '../data'
train_files, train_targets = load_dataset(data_folder_path+'/dogimages/train')
valid_files, valid_targets = load_dataset(data_folder_path+'/dogimages/valid')
test_files, test_targets = load_dataset(data_folder_path+'/dogimages/test')
dog_names = [item[35:-1] 
for item in sorted(glob(data_folder_path+'/dog_images/train/*/'))]

def path_to_tensor(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
	list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
	return np.vstack(list_of_tensors)


ImageFile.LOAD_TRUNCATED_IMAGES = True
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


def build_model():
	model = Sequential()
	
	model.add(Conv2D(filters=16, kernel_size=2, activation='relu', input_shape=(224, 224, 3)))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
	model.add(MaxPooling2D(pool_size=2))
	model.add(GlobalAveragePooling2D())
	model.add(Dense(units=133, activation='softmax'))
	
	return model



model = build_model()
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)

epochs = 200
model.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets),callbacks=[checkpointer],epochs=epochs, batch_size=20, verbose=1)

model.load_weights('saved_models/weights.best.from_scratch.hdf5')
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print("**********  Evaluate model ********** \n")
scores = model.evaluate(X_test, y_test, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))
print('Test accuracy: %.4f%%' % test_accuracy)
