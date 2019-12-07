import sys
from os import listdir
from os.path import isfile, join

import streamlit as st
import train_classifier
from PIL import Image, ImageFilter
from keras.models import load_model

if __name__ == '__main__':

    showpred = 0
    try:
        model_path = 'my_model.h5'
    except:
        print("Need to train model")
        sys.exit(0)

    test_path = 'dataset/validation/nsfw/'

    # Load the pre-trained models
    model = load_model(model_path)
    st.sidebar.title("About")

    st.sidebar.info(
        "This is demo identifies classifies picture as SFW or NSFW using ML, all the photos from this demo have been downloaded from Reddit.")

    onlyfiles = [f for f in listdir("dataset/validation/nsfw") if isfile(join("dataset/validation/nsfw", f))]

    st.sidebar.title("Train Neural Network")
    if st.sidebar.button('Train'):
        train_classifier.train().save('my_model.h5')

    st.sidebar.title("Predict New Images")
    imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)

    if st.sidebar.button('Predict'):
        showpred = 1
        prediction = train_classifier.predict(model, test_path + imageselect)

        st.title('NSFW Classification')
        st.write("Pick an image from the left. You'll be able to view the image.")
        st.write("When you're ready, submit a prediction on the left.")

    st.write("")
    image = Image.open(test_path + imageselect)

    st.image(image.filter(ImageFilter.BoxBlur(20)), use_column_width=True)

    if showpred == 1:
        if prediction == 0:
            st.write("NSFW")
        if prediction == 1:
            st.write("SFW")
