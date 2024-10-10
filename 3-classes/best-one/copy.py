import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob as gb

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Rescaling, InputLayer
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.applications.mobilenet import MobileNet, preprocess_input  # https://keras.io/api/applications/
from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import random_zoom, flip_axis
from keras.utils import image_dataset_from_directory
import tensorflow as tf

# train_dir = '/Users/andrey/kaggle/input/Skin cancer ISIC The International Skin Imaging Collaboration/Train'
# test_dir = '/Users/andrey/kaggle/input/Skin cancer ISIC The International Skin Imaging Collaboration/Test'

train_dir = '/Users/andrey/kaggle/input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/train'
valid_dir = '/Users/andrey/kaggle/input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/valid'
test_dir = '/Users/andrey/kaggle/input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/test'


# training = tf.keras.preprocessing.image.ImageDataGenerator(
#     zca_epsilon=1e-06,
#     rotation_range=30,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     fill_mode="nearest",
#     horizontal_flip=True,
#     vertical_flip=True,
#     preprocessing_function=preprocess_input,
#     validation_split=0.9
# ).flow_from_directory(train_dir, batch_size=128, target_size=(224, 224), subset="training")
#
# validating = tf.keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     fill_mode="nearest",
#     horizontal_flip=True,
#     vertical_flip=True,
#     preprocessing_function=preprocess_input,
#     validation_split=0.75
# ).flow_from_directory(valid_dir, batch_size=128, target_size=(224, 224), subset="validation")
#
# testing = tf.keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     fill_mode="nearest",
#     horizontal_flip=True,
#     vertical_flip=True,
#     preprocessing_function=preprocess_input,
#     validation_split=0.5
# ).flow_from_directory(test_dir, batch_size=128, target_size=(224, 224), subset="validation")


def preprocess(images, labels):
    return preprocess_input(images, mode="tf"), labels


def get_dataset_and_preprocess(d, p=None, mode="training"):
    if p is not None:
        ds = image_dataset_from_directory(
            d,
            image_size=(224, 224),
            label_mode='categorical',
            # validation_split=0.9,
            # subset="training",
            seed=125215215,
            batch_size=128,
            validation_split=p,
            subset=mode
        )

        return ds.map(preprocess)

    ds = image_dataset_from_directory(
        d,
        image_size=(224, 224),
        label_mode='categorical',
        # validation_split=0.9,
        # subset="training",
        seed=125215215,
        batch_size=128,
    )

    return ds.map(preprocess)


# training = image_dataset_from_directory(
#     train_dir,
#     image_size=(200, 200),
#     # validation_split=0.9,
#     # subset="training",
#     seed=125215215,
#     batch_size=64,
# )
#
# validating = image_dataset_from_directory(
#     valid_dir,
#     image_size=(200, 200),
#     # validation_split=0.75,
#     # subset="validation",
#     seed=125215215,
#     batch_size=64
# )
#
# testing = image_dataset_from_directory(
#     test_dir,
#     image_size=(200, 200),
#     seed=125215215,
#     batch_size=64
# )

training = get_dataset_and_preprocess(train_dir)
validating = get_dataset_and_preprocess(valid_dir)
testing = get_dataset_and_preprocess(test_dir)

mobilenet = VGG16(include_top=False, weights='imagenet',
                  input_shape=(224, 224, 3))  # https://keras.io/api/applications/

optimizer = SGD(learning_rate=0.001, momentum=0.1, nesterov=True)

EarlyStop = EarlyStopping(patience=5, restore_best_weights=True)
Reduce_LR = ReduceLROnPlateau(monitor='val_categorical_accuracy', verbose=2, factor=0.5, min_lr=0.00001)
callback = [EarlyStop, Reduce_LR]
# model_check=ModelCheckpoint('model.hdf5',monitor='val_loss',verbose=1,save_best_only=True)
# callback=[EarlyStop , Reduce_LR,model_check]

mobilenet.trainable = False

model = Sequential([
    mobilenet,
    # MaxPooling2D(3, 2),
    # Conv2D(kernel_size=3, filters=32, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.summary()

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["categorical_accuracy"])

history = model.fit(training, validation_data=validating, epochs=15, batch_size=128,
                    verbose=1)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

print(f"Test result: {model.evaluate(testing, return_dict=True)['categorical_accuracy']}")
