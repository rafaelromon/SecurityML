from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from mlxtend.plotting import plot_confusion_matrix


# TODO
#     Batch normalization
#     Dropout
#     Image augmentation
#     Confusion matrix
#
from tensorflow_core.python.keras.models import load_model


def train():
    matplotlib.use('TkAgg')

    PATH = "dataset"

    batch_size = 128
    epochs = 15
    IMG_HEIGHT = 300
    IMG_WIDTH = 300

    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')

    train_sfw_dir = os.path.join(train_dir, 'sfw')  # directory with our training sfw pictures
    train_nsfw_dir = os.path.join(train_dir, 'nsfw')  # directory with our training nsfw pictures
    validation_sfw_dir = os.path.join(validation_dir, 'sfw')  # directory with our validation sfw pictures
    validation_nsfw_dir = os.path.join(validation_dir, 'nsfw')  # directory with our validation nsfw pictures

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

    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )

    train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')

    sample_training_images, _ = next(train_data_gen)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

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

    model.save('my_model.h5')

    return model


def plot(model):
    img_path = os.path.join('dataset/validation')

    nsfw = 0
    sfw = 0
    should_be = 0
    false_positives = 0
    false_negatives = 0

    for subdir, dirs, files in os.walk(img_path):
        if "nsfw" in subdir or "sfw" in subdir:
            if "sfw" in subdir:
                should_be = 1
            else:
                should_be = 0

            for file in files:
                path_to = os.path.join(subdir, file)
                img = image.load_img(path_to, target_size=(300, 300), grayscale=False)
                img_tensor = image.img_to_array(img)
                img_tensor = np.expand_dims(img_tensor, axis=0)
                img_tensor /= 255.

                res = model.predict(img_tensor)

                if res > 0.5:
                    if should_be == 0:
                        false_positives += 1
                        sfw += 1
                    else:
                        if should_be == 1:
                            false_negatives += 1
                            nsfw += 1

    total_nsfw = nsfw + false_positives - false_negatives
    total_psfw = sfw + false_negatives - false_positives

    # Confusion matrix
    cm = np.array([[nsfw - false_negatives, false_positives], [false_negatives, sfw - false_positives]])
    plt.figure()
    plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True)
    plt.xticks(range(2), ['NSFW', 'SFW'], fontsize=16)
    plt.yticks(range(2), ['NSFW', 'SFW'], fontsize=16)
    plt.show()


if __name__ == '__main__':
    # model = train()
    model = load_model("my_model.h5")
    plot(model)
