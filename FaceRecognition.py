import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle

# Load the pickled faces list
with open('faces.pkl', 'rb') as f:
    faces_data = pickle.load(f)

# Streamlit interface
def main():
    st.title("Face Detection App")

    # Instructions for users
    st.write("Upload an image and the app will detect faces.")
    st.write("You can adjust the detection parameters and save the resulting image.")

    # Upload image
    image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if image_file:
        # Convert the uploaded file to OpenCV format
        image = Image.open(image_file)
        image = np.array(image)

        # Convert image to BGR format (OpenCV default)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the original image
        st.image(image_file, caption='Uploaded Image', use_column_width=True)

        # Face detection settings
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Sliders for adjusting parameters
        scale_factor = st.slider("Adjust scaleFactor", 1.1, 2.0, 1.3, step=0.1)
        min_neighbors = st.slider("Adjust minNeighbors", 3, 10, 5, step=1)

        # Color picker for the rectangle
        rect_color = st.color_picker("Pick a color for the rectangle", "#00FF00")
        rect_color = tuple(int(rect_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))  # Convert hex to RGB
        rect_color = (rect_color[2], rect_color[1], rect_color[0])  # Convert RGB to BGR for OpenCV

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), rect_color, 2)

        # Convert back to RGB for displaying in Streamlit
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image with detected faces
        st.image(image_rgb, caption="Detected Faces", use_column_width=True)

        # Button to save the image with detected faces
        if st.button("Save Image with Detected Faces"):
            save_path = "detected_faces.png"
            cv2.imwrite(save_path, image)
            st.write(f"Image saved as {save_path}")
            st.download_button(
                label="Download Image",
                data=open(save_path, "rb").read(),
                file_name="detected_faces.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
