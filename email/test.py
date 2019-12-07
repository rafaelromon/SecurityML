import io
import os
import re

import pandas as pd
import tensorflow as tf

regex = re.compile('[!?&�\"\\\]')

path_model = os.path.join('results/spam.h5')
model = tf.keras.models.load_model(path_model, compile=False)

e = model.layers[0]
weights = e.get_weights()[0]

df = pd.read_csv('Dataset/spam.csv', delimiter=';', encoding='latin-1')
mail = df.v2
words = set()
for e in mail:
    for w in e.split():
        words.add(w)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(words):
    vec = weights[num + 1]  # skip 0, it's padding.
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()
out_m.close()
