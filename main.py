import keras
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob as gb

# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Rescaling, InputLayer, \
#     RandomFlip, RandomZoom, RandomCrop
# from keras.optimizers import Adam, RMSprop, SGD
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from keras.applications.mobilenet import MobileNet, preprocess_input  # https://keras.io/api/applications/
# from keras.applications import MobileNetV2
# from keras.applications.imagenet_utils import preprocess_input
# from keras.preprocessing.image import random_zoom, flip_axis
# from keras.utils import image_dataset_from_directory


# train_dir = '/Users/andrey/kaggle/input/Skin cancer ISIC The International Skin Imaging Collaboration/Train'
# test_dir = '/Users/andrey/kaggle/input/Skin cancer ISIC The International Skin Imaging Collaboration/Test'

# train_dir = '/Users/andrey/kaggle/input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/train'
# valid_dir = '/Users/andrey/kaggle/input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/valid'
# test_dir = '/Users/andrey/kaggle/input/skin-lesion-analysis-toward-melanoma-detection/skin-lesions/test'


train_dir = '/Users/andrey/kaggle/input/dataset/train'
test_dir = '/Users/andrey/kaggle/input/dataset/test'

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


augmentation = keras.models.Sequential([
    keras.layers.RandomFlip(),
    keras.layers.RandomZoom(0.8, 0.7),
])


def preprocess(images, labels):
    return keras.applications.imagenet_utils.preprocess_input(images, mode="tf"), labels


def get_dataset_and_preprocess(d, p=None, mode="training"):
    if p is not None:
        ds = keras.utils.image_dataset_from_directory(
            d,
            image_size=(224, 224),
            label_mode='categorical',
            # validation_split=0.9,
            # subset="training",
            seed=125215215,
            batch_size=32,
            validation_split=p,
            subset=mode
        )

        return ds

    ds = keras.utils.image_dataset_from_directory(
        d,
        image_size=(224, 224),
        label_mode='categorical',
        # validation_split=0.9,
        # subset="training",
        seed=125215215,
        batch_size=32,
    )

    return ds


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

training = get_dataset_and_preprocess(train_dir, 0.1, "training")
validating = get_dataset_and_preprocess(train_dir, 0.1, "validation")
# (training, validating) = get_dataset_and_preprocess(train_dir, 0.1, "both")
testing = get_dataset_and_preprocess(test_dir)

# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),name='vgg16'))
# model.add(keras.layers.Flatten(name='flatten'))

# mobilenet = keras.applications.VGG16(include_top=False, weights='imagenet',
#                         input_shape=(224, 224, 3))  # https://keras.io/api/applications/

vgg = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg.trainable = False

model = keras.models.Sequential([
    keras.layers.InputLayer(shape=(224, 224, 3), batch_size=32),
    keras.layers.Rescaling(1. / 255),
    keras.layers.RandomZoom(height_factor=(.2, .4), width_factor=(.2, .4)),
    keras.layers.RandomFlip(),
    keras.layers.RandomRotation(factor=(-.4, .4)),
    # keras.layers.Conv2D(8, 3, padding='valid', activation='relu'),
    # keras.layers.Conv2D(8, 3, padding='valid', activation='relu'),
    # keras.layers.AvgPool2D(pool_size=(2, 2)),
    # keras.layers.Conv2D(16, 3, padding='valid', activation='relu'),
    # keras.layers.Conv2D(16, 3, padding='valid', activation='relu'),
    # keras.layers.MaxPooling2D(),
    # keras.layers.Conv2D(32, 3, padding='valid', activation='relu'),
    # keras.layers.Conv2D(32, 3, padding='valid', activation='relu'),
    # keras.layers.Conv2D(32, 3, padding='valid', activation='relu'),
    # keras.layers.MaxPooling2D(),
    # keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    vgg,
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
# mobilenet.trainable = False
# for layer in mobilenet.layers[:-4]:
#     layer.trainable = False

optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.8)

EarlyStop = keras.callbacks.EarlyStopping(restore_best_weights=True)
Reduce_LR = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', verbose=2, factor=0.5)
callback = [EarlyStop, Reduce_LR]
# model_check=ModelCheckpoint('model.hdf5',monitor='val_loss',verbose=1,save_best_only=True)
# callback=[EarlyStop , Reduce_LR,model_check]

# for layer in mobilenet.layers[:-20]:
#     layer.trainable = False

# model = keras.models.Sequential([
#     # augmentation,
#     mobilenet,
#     # MaxPooling2D(3, 2),
#     # Conv2D(kernel_size=3, filters=32, activation='relu'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dropout(0.6),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(2, activation='softmax')
# ])

model.summary()

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

history = model.fit(training, validation_data=validating, epochs=20, callbacks=callback,
                    verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# plt.plot(epochs, precision, 'b', label='Training precision')
# plt.plot(epochs, val_precision, 'r', label='Validation precision')
# plt.title('Training and validation precision')
# plt.legend()

# plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()

evals = model.evaluate(testing, return_dict=True)

print(f"Test precision: {evals['precision']}")
print(f"Test accuracy: {evals['accuracy']}")

model.save("saved_model.keras")
