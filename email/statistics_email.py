import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
import io
import pickle
from keras.backend import manual_variable_initialization
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.keras.utils import plot_model


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

manual_variable_initialization(True)
dir = os.path.join('Dataset')
data = pd.read_csv('Dataset/spam_or_not_spam.csv', delimiter=',')


mail = data.email
mail = mail.astype(str)
label = data.label
le = LabelEncoder()
label = le.fit_transform(label)
label = label.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(mail, label, test_size=0.99)

objects = ('HAM', 'SPAM')
y_pos = np.arange(len(objects))
performance = [list(label).count(0), list(label).count(1)]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Cantidad')
plt.title('Dataset split')

plt.show()

vocab = set()
for e in mail:
    for w in e.split():
        vocab.add(w)

max_words = len(vocab) # Vocab max size
max_len = 100 # Sentences padded to 100 words vector
with open(os.path.join('results/tokenizer.pickle'), 'rb') as handle:
    tok = pickle.load(handle)

tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

path_model = os.path.join("results/spam.h5")
model = tf.keras.models.load_model(path_model, compile=False)

Y_pred = model.predict(np.array(test_sequences_matrix))
y_pred = np.around(Y_pred)
cm = confusion_matrix(Y_test, y_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True)

plt.xticks(range(2), ['HAM', 'SPAM'], fontsize=16)
plt.yticks(range(2), ['HAM', 'SPAM'], fontsize=16)
# plt.savefig("matrix.png")
plt.show()