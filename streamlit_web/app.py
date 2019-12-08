import os
import pickle
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter
from encoder import encode_pe
from keras.backend import manual_variable_initialization
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences


def predict(model1, file):
    img_width, img_height = 224, 224
    x = load_img(file, target_size=(img_width, img_height), grayscale=True)
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model1.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer


manual_variable_initialization(True)

path_model_malware = os.path.join("models/malware.h5")
path_model_email = os.path.join("models/spam.h5")
path_model_sms = os.path.join("models/sms.h5")
path_model_nsfw = os.path.join("models/nsfw.h5")
model_malware = tf.keras.models.load_model(path_model_malware, compile=False)
model_email = tf.keras.models.load_model(path_model_email, compile=False)
model_sms = tf.keras.models.load_model("../streamlit_web/models/sms.h5", compile=False,
                                       custom_objects={"softmax_v2": tf.nn.softmax})
model_nsfw = tf.keras.models.load_model(path_model_nsfw, compile=False)

max_len = 100
# loading
with open(os.path.join('tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

df = pd.DataFrame({
    'first column': ["Email", "Malware", "NSFW", "SMS"]
})

option = st.sidebar.selectbox(
    'Select which application do you wanna test',
    df['first column'])

if option == "Malware":
    st.title('Malware detection')
    st.text("This model has been trained with a dataset with over 200000 .exe "
            "samples,\n validated with www.virustotal.com.")

    filename = st.text_input('Enter a file path:')
    try:
        encoded = encode_pe(os.path.join(filename))
        encoded = [float(x) for x in encoded]
        result = model_malware.predict(np.array([encoded]))[0][0]
        if round(result) == 1:
            st.text("Your file is malware with a probability of %.2f" % float(result))
        else:
            st.text("Your file is safe with a probability of %.2f" % float(1 - result))
    except FileNotFoundError:
        st.error('File not found.')

elif option == "Email":
    st.title('Email spam detection')

    st.text("This model has been trained with a dataset with over 5000 emails")

    email = st.text_input('Enter a sample email')
    tokenized = tokenizer.texts_to_sequences([email])
    sequence_padded = pad_sequences(tokenized, maxlen=max_len)
    try:
        result = model_email.predict(sequence_padded)[0][0]
        if round(result) == 1:
            st.text("Your email is spam with a probability of %.2f" % float(result))
        else:
            st.text("Your email is fine with a probability of %.2f" % float(1 - result))
    except FileNotFoundError:
        st.error('Input error')

elif option == "NSFW":

    showpred = 0
    test_path = 'dataset/validation/nsfw/'
    st.sidebar.info(
        "This is demo identifies classifies picture as SFW or NSFW using ML, all the photos from this demo have been downloaded from Reddit.")

    onlyfiles = [f for f in listdir("dataset/validation/nsfw") if isfile(join("dataset/validation/nsfw", f))]

    st.sidebar.title("Predict New Images")
    imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)

    if st.sidebar.button('Predict'):
        showpred = 1
        prediction = predict(model_nsfw, test_path + imageselect)

        st.title('NSFW Classification')
        st.write("Pick an image from the left. You'll be able to view the image.")
        st.write("When you're ready, submit a prediction on the left.")

    st.write("")
    image = Image.open(test_path + imageselect)

    st.image(image.filter(ImageFilter.BoxBlur(20)), use_column_width=True)
    if showpred == 1:
        if prediction == 0:
            st.write("NSFW")
        elif prediction == 1:
            st.write("SFW")

elif option == "Email":
    st.title('Email spam detection')

    st.text("This model has been trained with a dataset with over 3000 sms")

    sms = st.text_input('Enter a sample email')
    tokenized = tokenizer.texts_to_sequences([sms])
    sequence_padded = pad_sequences(tokenized, maxlen=max_len)

    try:
        result = model_email.predict(sequence_padded)[0][0]
        if round(result) == 1:
            st.text("Your sms is spam with a probability of %.2f" % float(result))
        else:
            st.text("Your sms is fine with a probability of %.2f" % float(1 - result))
    except FileNotFoundError:
        st.error('Input error')
