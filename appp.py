import pywt
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import json
import os

# Define paths
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model.pkl')
CLASS_DICT_PATH = 'class_dictionary.json'

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please make sure the file exists.")
    st.stop()

# Check if class dictionary file exists
if not os.path.exists(CLASS_DICT_PATH):
    st.error(f"Class dictionary file '{CLASS_DICT_PATH}' not found. Please make sure the file exists.")
    st.stop()

# Load the trained model
clf = joblib.load(MODEL_PATH)

# Load the class dictionary
with open(CLASS_DICT_PATH, "r") as f:
    class_dict = json.load(f)

# Function to perform face detection and recognition
def detect_and_recognize_faces(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # Perform face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Perform face recognition
    for (x, y, w, h) in faces:
        roi_color = image[y:y+h, x:x+w]
        resized_roi_color = cv2.resize(roi_color, (32, 32))
        img_har = w2d(np.array(resized_roi_color))
        scalled_raw_img = cv2.resize(img_har, (32, 32))

        # Ensure that the dimensions are consistent
        if scalled_raw_img.shape[0] * scalled_raw_img.shape[1] == 1024:
            scalled_raw_img = scalled_raw_img.flatten().reshape(1, -1)
        else:
            st.error("Dimensions of the input data are not as expected.")
            st.stop()

        # Predict labels
        result_raw = clf.predict(scalled_raw_img)
        label = list(class_dict.keys())[list(class_dict.values()).index(result_raw[0])]
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

# Function to perform wavelet transform
def w2d(img, mode='haar', level=1):
    imArray = img
    # Convert to float
    imArray = cv2.resize(imArray, (32, 32))
    imArray = np.float32(imArray)   
    imArray /= 255

    # Compute coefficients 
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)  
    coeffs_H[0] *= 0  

    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H

def main():
    st.title("Celebrity Face Detection and Recognition")
    st.sidebar.title("Options")
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        # Read the image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform face detection and recognition
        if st.sidebar.button('Detect Faces'):
            result_image = detect_and_recognize_faces(np.array(image))
            st.image(result_image, caption='Detected Faces', use_column_width=True)

if __name__ == '__main__':
    main()