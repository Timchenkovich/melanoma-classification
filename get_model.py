import argparse

import keras
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 64

def get_model():
  base_model = keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

  model = keras.models.Sequential([
      keras.layers.InputLayer(input_shape=(224, 224, 3), batch_size=batch_size),
      keras.layers.Rescaling(1. / 255),
      keras.layers.RandomZoom(height_factor=(.2, .4), width_factor=(.2, .4)),
      keras.layers.RandomFlip(),
      keras.layers.RandomRotation(factor=(-.4, .4)),
      base_model,
      keras.layers.Flatten(),
      keras.layers.BatchNormalization(),
      keras.layers.Dense(256, activation='relu'),
      keras.layers.BatchNormalization(),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.BatchNormalization(),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.BatchNormalization(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(1, activation='sigmoid')
  ])

  optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.8)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
  return model

