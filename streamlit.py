import streamlit as st
from tensorflow import keras
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


# Load the model
model = load_model("converted_keras\keras_model.h5", compile=False)

# Define the class names
class_names = ["general waste", "infectious waste", "pathological waste", "sharp waste"]

def predict_image(image):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Resize and crop the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

# Streamlit app
st.title("Waste Classification with Keras")

uploaded_file = st.file_uploader("Choose an image of waste...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Waste Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, score = predict_image(image)
    st.write(f"Waste Type: {label.capitalize()}")
    st.write(f"Confidence Score: {score:.2f}")

    # Add some information about the waste type
    if label == "general waste":
        st.write("General waste includes non-hazardous, non-recyclable waste.")
    elif label == "infectious waste":
        st.write("Infectious waste may contain pathogens and requires special handling.")
    elif label == "pathological waste":
        st.write("Pathological waste includes human tissues, organs, or bodily fluids.")
    elif label == "sharp waste":
        st.write("Sharp waste includes needles, scalpels, and other objects that can cause cuts or punctures.")

    st.write("Please ensure proper disposal according to your local regulations.")