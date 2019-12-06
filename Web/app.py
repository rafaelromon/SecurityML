import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from encoder import encode_pe
import os
import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.backend import manual_variable_initialization

manual_variable_initialization(True)

path_model_malware = os.path.join("models/malware.h5")
path_model_email = os.path.join("models/spam.h5")
model_malware = tf.keras.models.load_model(path_model_malware, compile=False)
model_email = tf.keras.models.load_model(path_model_email, compile=False)
max_len=100
# loading
with open(os.path.join('tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

df = pd.DataFrame({
  'first column': ["Email", "Malware", "NSFW", "SMS"]
})


option = st.sidebar.selectbox(
    'Select which application do you wanna test',
     df['first column'])

if option=="Malware":
    st.title('Malware detection')
    st.text("This model has been trained with a dataset with over 200000 .exe "
            "samples,\n validated with www.virustotal.com.")

    filename = st.text_input('Enter a file path:')
    try:
        encoded = encode_pe(os.path.join(filename))
        encoded = [float(x) for x in encoded]
        result = model_malware.predict(np.array([encoded]))[0][0]
        if round(result)==1:
            st.text("Your file is malware with a probability of %.2f"%float(result))
        else:
            st.text("Your file is safe with a probability of %.2f"%float(1-result))
    except FileNotFoundError:
        st.error('File not found.')
elif option=="Email":
    st.title('Email spam detection')

    st.text("This model has been trained with a dataset with over 5000 emails")

    email = st.text_input('Enter a sample email')
    tokenized = tokenizer.texts_to_sequences([email])
    sequence_padded = pad_sequences(tokenized, maxlen=max_len)
    try:
        result = model_email.predict(sequence_padded)[0][0]
        if round(result)==1:
            st.text("Your email is spam with a probability of %.2f"%float(result))
        else:
            st.text("Your email is fine with a probability of %.2f"%float(1-result))
    except FileNotFoundError:
        st.error('Input error')