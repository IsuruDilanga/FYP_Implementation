import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, load_model

save_path = "/Users/isurudissanayake/Documents/IIT/4th_Year/FYP_NEW_Implementation/flaskProject/save_image"  # Change this to your desired location

# Function to save the uploaded image to a specific location
def save_uploaded_image(uploaded_image, save_path):
    with open(os.path.join(save_path, uploaded_image.name), "wb") as f:
        f.write(uploaded_image.getbuffer())
    return st.success("Image saved successfully!")


# Function to load the image, preprocess it, extract features, predict, and interpret the prediction
def predict_image(input_image_path, model_path):
    target_size = (224, 224)

    # Load the pre-trained VGG16 model with top layers included
    base_model = VGG16(weights='imagenet', include_top=True)

    # Create a new model that takes the input of VGG16 and outputs the desired layer/
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

    # Process the input image
    img = cv2.imread(input_image_path)
    img = cv2.resize(img, target_size)
    img = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

    # Extract features using the full VGG16 model
    features = model.predict(img)  # Use the new model

    # Reshape features to match the expected input shape of trained_model
    features_reshaped = np.reshape(features, (features.shape[0], 14, 14, 512))  # Reshape to (None, 14, 14, 512)

    # Load the trained model
    trained_model = load_model(model_path)

    # Predict ASD probability using the trained model and extracted features
    prediction = trained_model.predict(features_reshaped)[0][0]  # Access the first element for ASD probability

    # Interpret the prediction
    rounded_prediction = round(prediction, 2)
    if rounded_prediction > 0.5:
        return f"Predicted ASD with probability: {rounded_prediction:.2f}"
    else:
        return f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}"


st.write("""
# Empowering
# Understanding:
ASD and Emotion detection with 
explainable AI insights
""")

# Add an image upload button
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Button to save the uploaded image
    if st.button("Save Image"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_uploaded_image(uploaded_image, save_path)

    # Predict and interpret the uploaded image
    model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/VGG16/VGG16Model.h5'
    prediction_result = predict_image(os.path.join(save_path, uploaded_image.name), model_path)
    st.write(prediction_result)
