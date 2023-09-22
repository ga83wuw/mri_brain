import os
import random
import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils import *
from models import *

# Size of images
im_width = 256
im_height = 256

EPOCHS = 10
BATCH_SIZE = 32
learning_rate = 1e-4
smooth = 1e-5

### DATA ###

image_filenames_train = []

# local
mask_files = glob('./archive/lgg-mri-segmentation/kaggle_3m/*/*_mask*')

for i in mask_files:
    image_filenames_train.append(i.replace('_mask', ''))

print(f"Len of dataset: ", len(image_filenames_train))

df = pd.DataFrame(data = {'image_filenames_train': image_filenames_train, 'mask': mask_files})

df_train, df_test = train_test_split(df, test_size = 0.1)

# Further split this val and train
df_train, df_val = train_test_split(df_train, test_size = 0.2)

# Referring Code From: https://github.com/zhixuhao/unet/blob/master/data.py
seed = 42

def train_generator(data_frame, 
                    batch_size, 
                    augmentation_dict,
                    image_color_mode = "rgb",
                    mask_color_mode = "grayscale",
                    image_save_prefix = "image", 
                    mask_save_prefix = "mask",
                    save_to_dir = 42,
                    target_size = (256, 256),
                    seed = seed
                    ):
    """
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    """
    image_datagen = ImageDataGenerator(**augmentation_dict)
    mask_datagen = ImageDataGenerator(**augmentation_dict)

    image_generator = image_datagen.flow_from_dataframe(data_frame,
                                                        x_col = "image_filenames_train",
                                                        class_mode = None,
                                                        color_mode = image_color_mode,
                                                        target_size = target_size,
                                                        batch_size = batch_size,
                                                        save_to_dir = save_to_dir,
                                                        save_prefix = image_save_prefix,
                                                        seed = seed
                                                        )

    mask_generator = mask_datagen.flow_from_dataframe(data_frame,
                                                      x_col = "mask",
                                                      class_mode = None,
                                                      color_mode = mask_color_mode,
                                                      target_size = target_size,
                                                      batch_size = batch_size,
                                                      save_to_dir = save_to_dir,
                                                      save_prefix = mask_save_prefix,
                                                      seed = seed
                                                      )

    train_gen = zip(image_generator, mask_generator)
    
    # normalize & return tuple (img, mask)
    for (img, mask) in train_gen:
        
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        
        yield (img, mask)

train_generator_param = dict(rotation_range = 0.2,
                             width_shift_range = 0.05,
                             height_shift_range = 0.05,
                             shear_range = 0.05,
                             zoom_range = 0.05,
                             horizontal_flip = True,
                             fill_mode = 'nearest')

train_gen = train_generator(df_train, 
                            BATCH_SIZE,
                            train_generator_param,
                            target_size = (im_height, im_width))
    
test_gen = train_generator(df_val, 
                           BATCH_SIZE,
                           dict(),
                           target_size = (im_height, im_width))
    
###=========###

model = unet(input_size = (im_height, im_width, 3))
print(model.summary())

decay_rate = learning_rate / EPOCHS

opt = Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = decay_rate, amsgrad = False)

model.compile(optimizer = opt, loss = dice_coefficients_loss, metrics = ["binary_accuracy", iou, dice_coefficients])

callbacks = [ModelCheckpoint('unet.hdf5', verbose = 1, save_best_only = True)]

history = model.fit(train_gen,
                    steps_per_epoch = len(df_train) / BATCH_SIZE, 
                    epochs = EPOCHS, 
                    callbacks = callbacks,
                    validation_data = test_gen,
                    validation_steps = len(df_val) / BATCH_SIZE)

