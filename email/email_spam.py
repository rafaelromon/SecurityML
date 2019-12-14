import io
import os
import pickle
import csv
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import keras.backend as K
import numpy as np

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

K.manual_variable_initialization(True)
dir = os.path.join('Dataset')
data = pd.read_csv('Dataset/spam_or_not_spam.csv', delimiter=',')


mail = data.email
mail = mail.astype(str)
label = data.label
le = LabelEncoder()
label = le.fit_transform(label)
label = label.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(mail, label, test_size=0.2)

vocab = set()
for e in mail:
    for w in e.split():
        vocab.add(w)

max_words = len(vocab)  # Vocab max size
max_len = 100  # Sentences padded to 100 words vector
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

# saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
# exit()


model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_words, 50, input_length=max_len),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

csv_logger = CSVLogger('log.csv')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, mode='min', restore_best_weights=True)
mc = ModelCheckpoint('spam.h5', monitor='val_loss', mode='min', verbose=1)

model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['acc', recall_threshold(0.9)])

history = model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10,
                    validation_data=(test_sequences_matrix, Y_test),
                    verbose=2,
                    # callbacks=[csv_logger, mc, early_stop]
                    )


model.save('spam.h5')

# print(model.predict(test_sequences_matrix[:10]))

# path_model = os.path.join('spam.h5')
# model = tf.keras.models.load_model(path_model)
#
#
# print(Y_test[:20])
#
# print(model.predict(test_sequences_matrix[:20]))

# e = model.layers[0]
# weights = e.get_weights()[0]
#
# out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
# out_m = io.open('meta.tsv', 'w', encoding='utf-8')
#
# words = list(vocab)
# for num, word in enumerate(words):
#     vec = weights[num]
#     out_m.write(word + "\n")
#     out_v.write('\t'.join([str(x) for x in vec]) + "\n")
# out_v.close()
# out_m.close()

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