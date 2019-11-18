
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.text import Tokenizer

dir = os.path.join('Dataset')
training_set = pd.read_csv("Dataset/spam_test.csv", encoding='latin-1')
valid_set = pd.read_csv("Dataset/spam_train.csv", encoding='latin-1')

X_train = training_set.Mail
X_test = training_set.Mail
Y_train = training_set.Label
Y_test = training_set.Label

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_train = Y_train.reshape(-1,1)
Y_test = le.fit_transform(Y_test)
Y_test = Y_test.reshape(-1,1)
# w2vec
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

sequences_t = tok.texts_to_sequences(X_test)
sequences_matrix_t = sequence.pad_sequences(sequences_t,maxlen=max_len)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_words, 200, input_length=max_len),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

history = model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,
                    validation_data=(sequences_matrix_t, Y_test))





# model.save('first_spam.h5')


# Plot training and test acc and loss
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

