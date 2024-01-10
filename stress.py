#conda env10
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Load the labels
class_names = ["stressless", "stressful"]

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Set the title and subtitle
st.title("AI Stress Evaluation Demo")
st.subheader("การประเมินความเครียดจากสีหน้าโดยกล้องหรือภาพถ่าย")

# Define a function to get the prediction from an uploaded image
def predict_image(image_file):
    # Load the image
    image = Image.open(image_file).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, method=Image.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the image and prediction
    caption = f"<p style='font-size: 24px'>Class: {class_name} ({class_name.replace('stress', '')})\nConfidence Score: {confidence_score:.2f}</p>"
    st.image(image, caption=None)
    st.markdown(caption, unsafe_allow_html=True)
    if class_name == "stressless":
        st.write("Relax as usual")
    else:
        st.write("Consider exercise, vacation, entertainment, or consult your physician")

# Define a function to get the prediction from the camera
def predict_camera():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to 224x224 and normalize the pixel values
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        normalized_frame = (resized_frame.astype(np.float32) / 127.5) - 1

        # Load the frame into the array
        data[0] = normalized_frame

        # Predict using the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Display the frame and prediction
        st.image(rgb_frame, channels="RGB", caption=f"Class: {class_name} ({class_name.replace('stress', '')})\nConfidence Score: {confidence_score:.2f}")
        
        if class_name == "stressless":
            st.write("Relax as usual")
        else:
            st.write("Consider exercise, vacation, entertainment, or consult your physician")
        # Check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Define the main function
def main():
    # Add a file uploader to allow the user to choose an image file
    uploaded_file = st.file_uploader("Upload ภาพถ่าย", type=["jpg", "jpeg", "png"])

    # Add a radio button to allow the user to choose between using the camera or an uploaded image
    option = st.radio("โปรดเลือก", ("Camera", "Image"))

    # If the user chooses the 'Camera' option, call the 'predict_camera' function
    if option == "Camera":
        st.write("Press 'q' to quit")
        predict_camera()

    # If the user chooses the 'Image' option and has uploaded an image file, call the 'predict_image' function
    elif option == "Image" and uploaded_file is not None:
        predict_image(uploaded_file)

    # If the user chooses the 'Image' option but has not uploaded an image file, display an error message
   
    elif option == "Image" and uploaded_file is None:
        st.error("Please upload an image file.")

# Run the main function
if __name__ == "__main__":
    main()