import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def process_data(file):
    dataframe = pd.read_csv(file, encoding='iso8859')

    labels = list()
    for line in dataframe.v1:
        labels.append(0 if line == 'ham' else 1)
    texts = list()
    for line in dataframe.v2:
        texts.append(line)
    lengths = list()
    for text in texts:
        lengths.append(len(text.split()))
    maxlen = max(lengths)
    labels = np.array(labels)
    texts = np.array(texts)

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    tokenized_messages = tokenizer.texts_to_sequences(texts)
    padded_messages = tf.keras.preprocessing.sequence.pad_sequences(tokenized_messages, maxlen)
    onehot_labels = tf.keras.utils.to_categorical(labels, num_classes=2)

    X = padded_messages
    Y = onehot_labels

    print(X.shape)
    print(Y.shape)
    print('MESSAGE MAXLEN = {}'.format(maxlen))

    train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2)

    output_path = 'dataset/processed/'
    np.save('{}x.npy'.format(output_path), train_features)
    np.save('{}y.npy'.format(output_path), train_labels)
    np.save('{}test_x.npy'.format(output_path), test_features)
    np.save('{}test_y.npy'.format(output_path), test_labels)

    with open('dataset/processed/word_dict.json', 'w') as file:
        json.dump(tokenizer.word_index, file)

    print('Data processed.')


if __name__ == '__main__':
    process_data('dataset/raw/spam.csv')
