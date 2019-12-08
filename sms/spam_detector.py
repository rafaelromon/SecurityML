import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.engine.saving import load_model
from keras.layers import *

NUMBER_OF_CLASSES = 2
MAXLEN = 171
BATCH_SIZE = 100
EPOCHS = 100


def train():
    X = np.load('dataset/processed/x.npy')
    Y = np.load('dataset/processed/y.npy')
    test_X = np.load('dataset/processed/test_x.npy')
    test_Y = np.load('dataset/processed/test_y.npy')

    print(X.shape)
    print(Y.shape)
    print(test_X.shape)
    print(test_Y.shape)

    dropout_rate = 0.5

    input_shape = (MAXLEN,)

    # TODO error while loading model
    leaky_relu = keras.layers.LeakyReLU(alpha=0.3)

    model_scheme = [

        Reshape(input_shape=input_shape, target_shape=(MAXLEN, 1)),

        Conv1D(128, kernel_size=2, strides=1, kernel_regularizer='l1'),
        LeakyReLU(),
        MaxPooling1D(pool_size=2),

        Flatten(),

        Dense(64),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(NUMBER_OF_CLASSES, activation=tf.nn.softmax)

    ]

    model = keras.Sequential(model_scheme)

    model.compile(
        optimizer=optimizers.Adam(lr=0.0001),
        loss="categorical_crossentropy",
        metrics=['accuracy'],
    )
    model.summary()

    print(Y)

    history = model.fit(X, Y,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(test_X, test_Y)
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
    plt.savefig("history.png")
    plt.show()

    return model


if __name__ == '__main__':
    model = train()

    model.save('../streamlit_web/models/sms.h5')
    model = load_model("../streamlit_web/models/sms.h5",
                       custom_objects={"softmax_v2": tf.nn.softmax})

    # TODO plot confusion matrix
