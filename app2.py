import streamlit as st
from tensorflow import keras
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import random
import folium
from folium import Marker, Map

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("converted_keras/keras_model.h5", compile=False)

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

def display_circles(predicted_class, confidence):
    # Define colors for each class
    color_map = {
        "general waste": "black",
        "infectious waste": "yellow",
        "pathological waste": "red",
        "sharp waste": "white"
    }

    if confidence >= 0.6:  # Show circle only if confidence is 60% or higher
        color = color_map[predicted_class]
        st.markdown(f"""
            <div style="display: flex; justify-content: center;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background-color: {color}; border: 2px solid gray;"></div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.write("Not Trash Detected")
        st.markdown(""" 
            <div style="display: flex; justify-content: center;">
                <div style="width: 30px; height: 30px; border-radius: 50%; background-color: gray; opacity: 0.3;"></div>
            </div>
        """, unsafe_allow_html=True)

# Streamlit app title
st.title("SmartWaste Poineers App")
st.markdown("### Classify waste types efficiently using AI and visualize dustbins.")

# Add tabs for classification, dustbin simulation, and editing dustbins
tab1, tab2, tab3 = st.tabs(["Waste Classification", "Locate Dustbins", "Edit Dustbins"])

# Waste Classification Tab
with tab1:
    uploaded_file = st.file_uploader("Choose an image of waste...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # Resize image for display
        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        st.image(image, caption="Uploaded Waste Image", use_column_width='auto')
        st.write("Classifying...")
        label, score = predict_image(image)

        st.write(f"**Waste Type:** {label.capitalize()}")
        st.write(f"**Confidence Score:** {score:.2f}")

        # Add information about the waste type
        waste_info = {
            "general waste": "General waste includes non-hazardous, non-recyclable waste.",
            "infectious waste": "Infectious waste may contain pathogens and requires special handling.",
            "pathological waste": "Pathological waste includes human tissues, organs, or bodily fluids.",
            "sharp waste": "Sharp waste includes needles, scalpels, and other objects that can cause cuts or punctures."
        }
        
        st.write(waste_info[label])
        st.write("Please ensure proper disposal according to your local regulations.")

        # Display colored circles
        display_circles(label, score)

# Dustbin Simulation Tab
with tab2:
    st.markdown("### Visualize the dustbins in the area.")
    
    # Add a slider to control the marker radius
    radius = st.slider("Select Marker Radius (in pixels):", min_value=5, max_value=50, value=20)

    # Sample random dustbin data
    num_dustbins = st.number_input("Number of Dustbins:", min_value=1, value=5, step=1)  # Allow user to change number of dustbins
    dustbin_data = []

    for _ in range(num_dustbins):
        location = (random.uniform(-1.2925, -1.2915), random.uniform(36.8200, 36.8220))  # Random locations
        dustbin_type = random.choice(class_names)
        dustbin_data.append({"location": location, "type": dustbin_type})

    # Create a map centered on a random location
    m = Map(location=[-1.292, 36.821], zoom_start=15)

    # Prepare data for the map
    for db in dustbin_data:
        lat, lon = db["location"]
        dustbin_type = db["type"]
        
        # Define colors for markers
        color_map = {
            "general waste": "black",
            "infectious waste": "yellow",
            "pathological waste": "red",
            "sharp waste": "white"
        }
        color = color_map[dustbin_type]

        # Add a marker with a circle for radius
        Marker(
            location=[lat, lon],
            popup=f"{dustbin_type.capitalize()}",
            icon=None  # No default icon to use custom styling
        ).add_to(m)

    # Display the map
    st.components.v1.html(m._repr_html_(), width=700, height=500)

    # Display the dustbins with color-coded markers
    st.markdown("### Dustbin Locations with Color-Coded Markers:")
    for db in dustbin_data:
        color = color_map[db["type"]]
        st.markdown(f"""
            <div style="position: relative; display: flex; justify-content: center;">
                <div style="width: {radius}px; height: {radius}px; border-radius: 50%; background-color: {color}; border: 2px solid gray;"></div>
                <div style="margin-left: 10px; align-self: center;">{db["type"].capitalize()}</div>
            </div>
        """, unsafe_allow_html=True)

# Edit Dustbins Tab
with tab3:
    st.markdown("### Edit the number and types of dustbins.")

    # Input for number of dustbins
    num_dustbins = st.number_input("Number of Dustbins:", min_value=0, value=len(dustbin_data), step=1)

    # Initialize dustbin details
    dustbin_details = []
    
    # Form for dustbin details
    for i in range(num_dustbins):
        col1, col2, col3 = st.columns(3)
        with col1:
            dustbin_type = st.selectbox(f"Dustbin {i + 1} Type:", class_names, key=f"type_{i}")
        with col2:
            status = st.selectbox(f"Dustbin {i + 1} Status:", ["remaining", "full"], key=f"status_{i}")
        with col3:
            location_lat = st.number_input(f"Dustbin {i + 1} Latitude:", value=random.uniform(-1.2925, -1.2915), key=f"lat_{i}")
            location_lon = st.number_input(f"Dustbin {i + 1} Longitude:", value=random.uniform(36.8200, 36.8220), key=f"lon_{i}")
        
        dustbin_details.append({
            "type": dustbin_type,
            "status": status,
            "location": (location_lat, location_lon)
        })

    # Update button
    if st.button("Update Dustbins"):
        # Prepare updated dustbin data
        updated_dustbin_data = [{"location": db["location"], "type": db["type"], "status": db["status"]} for db in dustbin_details]
        
        # Prepare data for the map
        updated_locations = [db["location"] for db in updated_dustbin_data]
        updated_types = [db["type"] for db in updated_dustbin_data]

        # Create a new map centered on the updated locations
        m = Map(location=[-1.292, 36.821], zoom_start=15)

        for db in updated_dustbin_data:
            lat, lon = db["location"]
            dustbin_type = db["type"]

            # Add a marker for each updated dustbin
            Marker(
                location=[lat, lon],
                popup=f"{dustbin_type.capitalize()}",
                icon=None  # No default icon to use custom styling
            ).add_to(m)

        # Display the updated map
        st.components.v1.html(m._repr_html_(), width=700, height=500)

        st.success("Dustbins updated successfully!")

# Run the app using `streamlit run your_script.py`
