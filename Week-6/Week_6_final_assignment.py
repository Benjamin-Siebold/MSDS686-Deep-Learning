#!/usr/bin/env python
# coding: utf-8


#Import libraries

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import backend, models, layers, optimizers, regularizers
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from IPython.display import display # Library to help view images
from PIL import Image # Library to help view images
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Library for data augmentation
import os, shutil # Library for navigating files
np.random.seed(42)


# Specify the base directory where images are located.  You need to save your data here.
base_dir = '/storage/msds686/cats_and_dogs/data/'


# Specify the traning, validation, and test dirrectories.  
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


# Specify the the classess in the training, validataion, and test dirrectories
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')


# We need to normalize the pixels in the images.  The data will 'flow' through this generator.
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#set Epoch
epoch = 50

# Since the file images are in a dirrectory we need to move them from the dirrectory into the model.  
# Keras as a function that makes this easy. Documentaion is here: https://keras.io/preprocessing/image/

train_generator = train_datagen.flow_from_directory(
    train_dir, # The directory where the train data is located
    target_size=(150, 150), # Reshape the image to 150 by 150 pixels. This is important because it makes sure all images are the same size.
    batch_size=20, # We will take images in batches of 20.
    class_mode='binary') # The classification is binary.

validataion_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

train_datagen2 = ImageDataGenerator(
    rescale=1./255,# The image augmentaion function in Keras
    rotation_range=40, # Rotate the images randomly by 40 degrees
    width_shift_range=0.2, # Shift the image horizontally by 20%
    height_shift_range=0.2, # Shift the image veritcally by 20%
    zoom_range=0.2, # Zoom in on image by 20%
    horizontal_flip=True, # Flip image horizontally 
    fill_mode='nearest') # How to fill missing pixels after a augmentaion opperation


test_datagen2 = ImageDataGenerator(rescale=1./255) #Never apply data augmentation to test data. You only want to normalize and resize test data. 

train_generator2 = train_datagen2.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validataion_generator2 = train_datagen2.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_generator2 = test_datagen2.flow_from_directory( # Resize test data
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

from tensorflow.keras.applications import Xception

backend.clear_session()
base_model = Xception(weights = 'imagenet',
                      include_top = False,
                      input_shape = (150,150,3)
                      )

print('Xception base summary:', base_model.summary())

base_model.trainable = False

print('Xception post freeze:', base_model.summary())

modelXception = models.Sequential()
modelXception.add(base_model)
modelXception.add(layers.Flatten())
modelXception.add(layers.Dense(256, activation = 'relu'))
modelXception.add(layers.Dense(1, activation = 'sigmoid'))

modelXception.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

history = modelXception.fit_generator(
    train_generator2,
    steps_per_epoch = 200,
    epochs = epoch,
    validation_data = validataion_generator2,
    verbose = 2,
    callbacks = [EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True)]
)

test_loss, test_acc = modelXception.evaluate_generator(test_generator2, steps = 50)

print('Xception_frozen_test_acc:', test_acc)

print('Above 95%!')

## Model 2

backend.clear_session()

base_model2 = Xception(weights = 'imagenet', 
                        include_top = False,
                        input_shape = (150, 150, 3)
                        )

for layer in base_model2.layers[:-7]:
    layer.trainable = False
for layer in base_model2.layers:
    print(layer, layer.trainable)

print('Xception Model unfrozen:', base_model2.summary())

modelXception2 = models.Sequential()
modelXception2.add(base_model2)
modelXception2.add(layers.Flatten())
modelXception2.add(layers.Dense(512, activation = 'relu'))
modelXception2.add(layers.Dense(1, activation = 'sigmoid'))

modelXception2.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

history = modelXception2.fit_generator(
    train_generator2,
    steps_per_epoch = 200,
    epochs = epoch,
    validation_data = validataion_generator2,
    verbose = 2,
    callbacks = [EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True)]
)

test_loss, test_acc = modelXception2.evaluate_generator(test_generator2, steps = 50)

print('Xception_unfrozen_test_acc:', test_acc)

print('Above 97%!!')

## Last model

backend.clear_session()

base_model2 = Xception(weights = 'imagenet', 
                        include_top = False,
                        input_shape = (150, 150, 3)
                        )

for layer in base_model2.layers[:-7]:
    layer.trainable = False
for layer in base_model2.layers:
    print(layer, layer.trainable)

print('Xception Model unfrozen:', base_model2.summary())

modelXception2 = models.Sequential()
modelXception2.add(base_model2)
modelXception2.add(layers.Flatten())
modelXception2.add(layers.Dense(512, activation = 'relu'))
modelXception2.add(BatchNormalization())
modelXception2.add(Dropout(.2))
modelXception2.add(layers.Dense(256, activation = 'relu'))
modelXception2.add(layers.Dense(1, activation = 'sigmoid'))

modelXception2.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

history = modelXception2.fit_generator(
    train_generator2,
    steps_per_epoch = 200,
    epochs = epoch,
    validation_data = validataion_generator2,
    verbose = 2,
    callbacks = [EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True)]
)

test_loss, test_acc = modelXception2.evaluate_generator(test_generator2, steps = 50)

print('Xception_unfrozen_test_acc:', test_acc)

print('Adding normalization and an additional layer at the end actually slightly reduced the accuracy. For this reason it would be best to stick with the second iteration of this model')
