import argparse
parser = argparse.ArgumentParser("train")
parser.add_argument("dataset_path", help="Path to folder where train and test data is stored.")
parser.add_argument("-m", "--model-path", help="Path to trained model")
args = parser.parse_args()

PATH_TO_DATASET = args.dataset_path

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

if args.model_path:
    model = keras.saving.load_model(args.model_path)
else:
    from get_model import get_model
    model = get_model()

model.summary()

EarlyStop = keras.callbacks.EarlyStopping(start_from_epoch=10, restore_best_weights=True)
Reduce_LR = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', verbose=2, factor=0.5)
callback = [EarlyStop, Reduce_LR]

history = model.fit(training, validation_data=validating, epochs=7, callbacks=callback,
                    verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig("loss.png")

plt.figure()

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig("accuracy.png")

evals = model.evaluate(testing, return_dict=True)

print(f"Test accuracy: {evals['accuracy']}")

model.save("model.keras")
