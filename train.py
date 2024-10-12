import os
import argparse
parser = argparse.ArgumentParser("train")
parser.add_argument("dataset_path", help="Path to folder where train and test data is stored.")
parser.add_argument("-m", "--model-path", help="Path to trained model")
parser.add_argument("output_dir", help="Path to output dir")
args = parser.parse_args()

PATH_TO_DATASET = args.dataset_path
OUTPUT_DIR = args.output_dir
MODEL_PATH = args.model_path

if os.path.exists(OUTPUT_DIR):
    print("Output dir already exists")
    import sys
    sys.exit()

os.mkdir(OUTPUT_DIR)
train_log = open(f"{OUTPUT_DIR}/train.log", "a")
test_log = open(f"{OUTPUT_DIR}/test.log", "a")

import keras
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = f"{PATH_TO_DATASET}/train"
test_dir = f"{PATH_TO_DATASET}/test"
batch_size = 64

def get_dataset(d, p=None, mode="training"):
    if p is not None:
        ds = keras.utils.image_dataset_from_directory(
            d,
            image_size=(224, 224),
            label_mode='binary',
            seed=125215215,
            batch_size=batch_size,
            validation_split=p,
            subset=mode
        )

        return ds

    ds = keras.utils.image_dataset_from_directory(
        d,
        image_size=(224, 224),
        label_mode='binary',
        seed=125215215,
        batch_size=batch_size,
    )

    return ds


training = get_dataset(train_dir, 0.1, "training")
validating = get_dataset(train_dir, 0.1, "validation")
testing = get_dataset(test_dir)

if MODEL_PATH:
    model = keras.saving.load_model(MODEL_PATH)
else:
    from get_model import get_model
    model = get_model()

model.summary()

max_epochs = 10

ModelCheckpoint = keras.callbacks.ModelCheckpoint(filepath=f"{OUTPUT_DIR}/checkpoint.keras", monitor="loss", mode="min", save_best_only=True)
Reduce_LR = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', verbose=2, factor=0.5)
callbacks = [ModelCheckpoint, Reduce_LR]

for i in range(max_epochs):
  history = model.fit(training, validation_data=validating, callbacks=callbacks,
                    verbose=1)
  evals = model.evaluate(testing, return_dict=True)

  test_log.write(f"{evals['accuracy']}\n")
  train_log.write(f"{i} {history.history['accuracy'][0]} {history.history['val_accuracy'][0]}\n")
  test_log.flush()
  train_log.flush()

test_log.close()
train_log.close()

#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#loss = history.history['loss']
#val_loss = history.history['val_loss']

#epochs = range(len(acc))

#plt.plot(epochs, loss, 'b', label='Training Loss')
#plt.plot(epochs, val_loss, 'r', label='Validation Loss')
#plt.title('Training and validation loss')
#plt.legend()

#plt.savefig("loss.png")

#plt.figure()

#plt.plot(epochs, acc, 'b', label='Training Accuracy')
#plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
#plt.title('Training and validation accuracy')
#plt.legend()

#plt.savefig("accuracy.png")

#evals = model.evaluate(testing, return_dict=True)

#print(f"Test accuracy: {evals['accuracy']}")

#model.save("model.keras")
