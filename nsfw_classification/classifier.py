from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import streamlit as st

from tensorflow_core.python.keras.models import load_model

IMG_HEIGHT = 255
IMG_WIDTH = 255

BATCH_SIZE = 128

PATH = "dataset"
TRAIN_DIR = os.path.join(PATH, 'train')
VALIDATION_DIR = os.path.join(PATH, 'validation')
EPOCHS = 15


def train():
    train_sfw_dir = os.path.join(TRAIN_DIR, 'sfw')  # directory with our training sfw pictures
    train_nsfw_dir = os.path.join(TRAIN_DIR, 'nsfw')  # directory with our training nsfw pictures
    validation_sfw_dir = os.path.join(VALIDATION_DIR, 'sfw')  # directory with our validation sfw pictures
    validation_nsfw_dir = os.path.join(VALIDATION_DIR, 'nsfw')  # directory with our validation nsfw pictures

    num_sfw_tr = len(os.listdir(train_sfw_dir))
    num_nsfw_tr = len(os.listdir(train_nsfw_dir))

    num_sfw_val = len(os.listdir(validation_sfw_dir))
    num_nsfw_val = len(os.listdir(validation_nsfw_dir))

    total_train = num_sfw_tr + num_nsfw_tr
    total_val = num_sfw_val + num_nsfw_val

    print('total training sfw images:', num_sfw_tr)
    print('total training nsfw images:', num_nsfw_tr)

    print('total validation sfw images:', num_sfw_val)
    print('total validation nsfw images:', num_nsfw_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)

    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.2
    )

    train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=TRAIN_DIR,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         color_mode="grayscale",  # TODO cambiado para transfer
                                                         class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=VALIDATION_DIR,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  color_mode="grayscale",
                                                                  class_mode='binary')

    sample_training_images, _ = next(train_data_gen)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # model = Sequential({
    #     DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #     Flatten(),
    #     Dense(1, activation='sigmoid')
    # })

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # model.summary()

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=total_val // BATCH_SIZE
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.savefig("history.png")

    return model

def plot(model):
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(os.path.join(PATH, 'validation'),
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            batch_size=BATCH_SIZE,
                                                            color_mode="grayscale",
                                                            class_mode='binary')

    Y_pred = model.predict_generator(validation_generator, 249 // BATCH_SIZE + 1)
    y_pred = np.around(Y_pred)
    confusion_matrix(validation_generator.classes, y_pred)

    cm = confusion_matrix(validation_generator.classes, y_pred)
    plt.figure()
    plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True)
    plt.xticks(range(2), ['NSFW', 'SFW'], fontsize=16)
    plt.yticks(range(2), ['NSFW', 'SFW'], fontsize=16)
    plt.show()
    # st.map(plt)
    # plt.savefig("matrix.png")


if __name__ == '__main__':
    # st.title('NSFW Classifier')
    # matplotlib.use('TkAgg')

    model = train()
    model.save('my_model.h5')
    model = load_model("my_model.h5")
    plot(model)
