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

dir = os.path.join('Dataset')
df = pd.read_csv('Dataset/spam.csv',delimiter=';',encoding='latin-1')
mail = df.v2
label = df.v1
le = LabelEncoder()
label = le.fit_transform(label)
label = label.reshape(-1, 1)
X_train,X_test,Y_train,Y_test = train_test_split(mail, label, test_size=0.2)

vocab = set()
for e in mail:
    for w in e.split():
        vocab.add(w)
max_words = len(vocab) # Vocab max size
max_len = 100 # Sentences padded to 100 words vector
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

# saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
exit()
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_words, 100, input_length=max_len),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

csv_logger = CSVLogger('log.csv')
#min_delta = 0.1 - quiere decir que cada epoch debe mejorar un 0.1% por lo menos, vamos de 0.82 a 0.821
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, mode='min', restore_best_weights=True)
mc = ModelCheckpoint('spam.h5', monitor='val_loss', mode='min', verbose=1)
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['acc'])


history = model.fit(sequences_matrix,Y_train,batch_size=128,epochs=15,
                    validation_data=(test_sequences_matrix, Y_test),
                    callbacks=[csv_logger, mc, early_stop])


e = model.layers[0]
weights = e.get_weights()[0]

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

words = list(vocab)
for num, word in enumerate(words):
  vec = weights[num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()

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
